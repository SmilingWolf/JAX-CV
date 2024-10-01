# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils very specific to this project, not generic."""

import dataclasses
import re
from typing import Mapping

import flax
import jax
import numpy as np
from absl import logging

from .pp import registry as pp_registry

Registry = pp_registry.Registry


# pylint: disable=logging-fstring-interpolation


def _traverse_with_names(tree, with_inner_nodes=False):
    """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
    if dataclasses.is_dataclass(tree):
        tree = flax.serialization.to_state_dict(tree)
    # Don't output the non-leaf nodes. If the optimizer doesn't have a state
    # the tree leaves can be Nones which was interpreted as a leaf by this
    # function but not by the other functions (like jax.tree.map).
    if tree is None:
        return
    elif isinstance(tree, Mapping):
        keys = sorted(tree.keys())
        for key in keys:
            for path, v in _traverse_with_names(tree[key], with_inner_nodes):
                yield (key + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    elif isinstance(tree, (list, tuple)):
        for idx in range(len(tree)):
            for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
                yield (str(idx) + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    else:
        yield "", tree


def tree_flatten_with_names(tree):
    """Populates tree_flatten with leaf names.

    This function populates output of tree_flatten with leaf names, using a
    custom traversal that produces names is provided. The custom traversal does
    NOT have to traverse tree in the same order as jax, as we take care of
    automatically aligning jax' and custom traversals.

    Args:
      tree: python tree.

    Returns:
      A list of values with names: [(name, value), ...]
    """
    vals, tree_def = jax.tree.flatten(tree)

    # "Fake" token tree that is use to track jax internal tree traversal and
    # adjust our custom tree traversal to be compatible with it.
    tokens = range(len(vals))
    token_tree = tree_def.unflatten(tokens)
    val_names, perm = zip(*_traverse_with_names(token_tree))
    inv_perm = np.argsort(perm)

    # Custom traverasal should visit the same number of leaves.
    assert len(val_names) == len(vals)

    return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(f, tree, *rest):
    """Like jax.tree.map but with a filter on the leaf path name.

    Args:
      f: A function with first parameter `name` (path-like "a/b/c") and remaining
        parameters values of `tree` and `*rest` corresponding to the given `name`
        Should return a new value for parameter `name`.
      tree: The tree of parameters `f` should be applied to.
      *rest: more trees of the exact same structure.

    Returns:
      A tree identical in structure to `tree` and `*rest` but with the leaves the
      result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
    """
    names_and_vals, tree_def = tree_flatten_with_names(tree)
    names, vals = zip(*names_and_vals)
    rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
    vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
    return tree_def.unflatten(vals)


def check_and_compile_patterns(patterns):
    """Validates and compiles a list of param-patterns.

    The validation consists of checking for common mistakes, currently only that
    the pattern does not start with a slash, because unlike FLAX, our parameter
    names don't start with a slash.

    Args:
      patterns: a single (string) pattern (regex), or a list of patterns.

    Returns:
      A list of compiled and verified regexes.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    assert isinstance(patterns, (list, tuple)), patterns

    def check_and_compile(pattern):
        assert not pattern.startswith(
            "/"
        ), f"Big vision parameter names never start with '/': '{pattern}"
        return re.compile(pattern)

    return list(map(check_and_compile, patterns))


def make_mask_trees(tree, patterns, *, log=None):
    """Returns a boolean mask tree for every pattern (only first match)."""
    compiled_patterns = check_and_compile_patterns(patterns)

    def matchfirst(name, _):
        matches = []
        for pattern in compiled_patterns:
            matches.append(not any(matches) and bool(pattern.fullmatch(name)))
        if log is not None and True in matches and jax.process_index() == 0:
            logging.info(
                "%s: %s - matched by %s", log, name, patterns[matches.index(True)]
            )
        return np.array(matches)

    multimask = tree_map_with_names(matchfirst, tree)
    return [
        jax.tree.map(lambda matches, i=idx: matches[i], multimask)
        for idx in range(len(patterns))
    ]


def tree_broadcast(prefix, target):
    """Broadcasts a prefix tree to a full tree.

    Input-output examples:
    1. prefix: {"x": 10, "y": 20}
       target: {"x": {"a": 1, "b": 2}, "y": 3}

       Result: {"x": {"a": 10, "b": 10}, "y": 20}

    2. prefix: 100
       target: {"x": {"a": 1, "b": 2}, "y": 3}

       Result: {"x": {"a": 100, "b": 100}, "y": 100}

    3. prefix: {"x": 10}
       target: {"x": {"a": 1, "b": 2}, "y": 3}

       Result: ValueError

    Args:
      prefix: prefix pytree.
      target: boradcast target for a prefix tree.

    Returns:
      prefix tree broadcasted to a target tree.
    """

    def _broadcast(leaf, subtree):
        return jax.tree.map(lambda _: leaf, subtree)

    return jax.tree.map(_broadcast, prefix, target)


def reshard(tree, shardings):
    """Take an arbitrarily* sharded pytree and shard it according to `shardings`.

    This is a no-op for tree elements which are already sharded as requested.

    *Arrays that are fully addressable (for example, CPU arrays) are assumed to be
    identical (i.e. replicated) across hosts.

    *It does not work if an element of `tree` is not fully-addressable, unless its
    sharding is already consistent with the target sharding.
    If this is needed, please ping lbeyer@ or akolesnikov@.

    Args:
      tree: a pytree of arrays.
      shardings: a (prefix) pytree of jax array shardings.
    Returns:
      A pytree of global jax arrays that follows provided shardings.
    """

    def _make_global_arr(x, shard, shape):
        # Avoid unnecessary copies and transfers:
        if hasattr(x, "sharding") and x.sharding.is_equivalent_to(shard, len(shape)):
            return x
        if not getattr(x, "is_fully_addressable", True):
            raise RuntimeError(
                "Trying to reshard a non-fully-addressable array. "
                "Please see the doc-comment for detailed explanation."
            )
        x = jax.device_get(x)  # Might be on local devices.
        xs = [
            jax.device_put(x[s], device=d)
            for d, s in shard.addressable_devices_indices_map(shape).items()
        ]
        return jax.make_array_from_single_device_arrays(shape, shard, xs)

    shapes = jax.tree.map(np.shape, tree)
    shardings = tree_broadcast(shardings, tree)
    return jax.tree.map(_make_global_arr, tree, shardings, shapes)
