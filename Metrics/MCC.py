import flax
import jax.numpy as jnp
from clu import metrics


@flax.struct.dataclass
class MCC(metrics.Metric):
    """
    Computes the Matthews correlation coefficient
    from model outputs 'logits' and 'labels'.

    The used formula helps avoiding overflow
    https://leimao.github.io/blog/Matthews-Correlation-Coefficient/
    """

    @classmethod
    def with_config(
        cls,
        threshold: float,
        averaging: str,
        num_classes: int,
        from_logits: bool,
    ):
        @flax.struct.dataclass
        class WithConfig(cls):
            true_positives: jnp.array
            true_negatives: jnp.array
            false_positives: jnp.array
            false_negatives: jnp.array

            @classmethod
            def empty(cls):
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
            def from_model_output(cls, *, logits: jnp.array, labels: jnp.array, **_):
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

            def merge(self, other: metrics.Metric) -> metrics.Metric:
                return type(self)(
                    true_positives=self.true_positives + other.true_positives,
                    true_negatives=self.true_negatives + other.true_negatives,
                    false_positives=self.false_positives + other.false_positives,
                    false_negatives=self.false_negatives + other.false_negatives,
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
                denominator = jnp.power(denominator, -0.5)
                return jnp.mean(numerator * denominator)

        return WithConfig
