import flax
import jax.numpy as jnp
from clu import metrics


@flax.struct.dataclass
class ConfusionMatrix(metrics.Metric):
    true_positives: jnp.array
    true_negatives: jnp.array
    false_positives: jnp.array
    false_negatives: jnp.array

    @classmethod
    def empty(cls, averaging, num_classes):
        if averaging == "micro":
            shape = (1,)
        elif averaging == "macro":
            shape = (num_classes,)

        return cls(
            true_positives=jnp.zeros(shape, dtype=jnp.int32),
            true_negatives=jnp.zeros(shape, dtype=jnp.int32),
            false_positives=jnp.zeros(shape, dtype=jnp.int32),
            false_negatives=jnp.zeros(shape, dtype=jnp.int32),
        )

    @classmethod
    def from_model_output(
        cls,
        *,
        logits: jnp.array,
        labels: jnp.array,
        from_logits: bool,
        threshold: float,
        averaging: str,
        **_
    ):
        preds = logits

        if from_logits:
            preds = flax.linen.activation.sigmoid(preds)

        labels = labels > threshold
        preds = preds > threshold

        if averaging == "micro":
            axis = None
        elif averaging == "macro":
            axis = 0

        return cls(
            true_positives=((preds == 1) & (labels == 1)).sum(axis=axis),
            true_negatives=((preds == 0) & (labels == 0)).sum(axis=axis),
            false_positives=((preds == 1) & (labels == 0)).sum(axis=axis),
            false_negatives=((preds == 0) & (labels == 1)).sum(axis=axis),
        )

    def merge(self, other: metrics.Metric):
        return type(self)(
            true_positives=self.true_positives + other.true_positives,
            true_negatives=self.true_negatives + other.true_negatives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )

    def compute(self):
        print("Must override compute()")
        raise NotImplementedError


def mcc(threshold, num_classes, from_logits, averaging):
    @flax.struct.dataclass
    class MCC(ConfusionMatrix):
        """
        Computes the Matthews correlation coefficient
        from model outputs 'logits' and 'labels'.

        The used formula helps avoiding overflow
        https://leimao.github.io/blog/Matthews-Correlation-Coefficient/
        """

        @classmethod
        def empty(cls):
            return super().empty(averaging, num_classes)

        @classmethod
        def from_model_output(cls, *, logits: jnp.array, labels: jnp.array, **_):
            return super().from_model_output(
                logits=logits,
                labels=labels,
                from_logits=from_logits,
                threshold=threshold,
                averaging=averaging,
            )

        def compute(self):
            N = (
                self.true_positives
                + self.false_negatives
                + self.false_positives
                + self.true_negatives
            )
            S = (self.true_positives + self.false_negatives) / N
            P = (self.true_positives + self.false_positives) / N
            numerator = (self.true_positives / N) - (S * P)
            denominator = S * P * (1 - S) * (1 - P)
            denominator = jnp.maximum(denominator, 1e-12)
            denominator = jnp.sqrt(denominator)
            return jnp.mean(numerator / denominator)

    return MCC


def f1score(threshold, num_classes, from_logits, averaging):
    @flax.struct.dataclass
    class F1Score(ConfusionMatrix):
        """
        Computes the F1 score
        from model outputs 'logits' and 'labels'.
        """

        @classmethod
        def empty(cls):
            return super().empty(averaging, num_classes)

        @classmethod
        def from_model_output(cls, *, logits: jnp.array, labels: jnp.array, **_):
            return super().from_model_output(
                logits=logits,
                labels=labels,
                from_logits=from_logits,
                threshold=threshold,
                averaging=averaging,
            )

        def compute(self):
            numerator = 2 * self.true_positives
            denominator = (
                (2 * self.true_positives) + self.false_positives + self.false_negatives
            )

            idx = jnp.where(denominator == 0)
            numerator = numerator.at[idx].set(1)
            denominator = denominator.at[idx].set(1)

            return jnp.mean(numerator / denominator)

    return F1Score
