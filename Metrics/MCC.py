import flax
import jax
import jax.numpy as jnp
from clu import metrics


@flax.struct.dataclass
class MCC(metrics.Metric):
    """
    Computes the micro-averaged Matthews correlation coefficient
    from model outputs 'logits' and 'labels'.

    The used formula helps avoiding overflow
    https://leimao.github.io/blog/Matthews-Correlation-Coefficient/
    """

    @classmethod
    def with_config(cls, threshold: float, from_logits: bool):
        @flax.struct.dataclass
        class WithConfig(cls):
            true_positives: jnp.array
            true_negatives: jnp.array
            false_positives: jnp.array
            false_negatives: jnp.array

            @classmethod
            def empty(cls):
                return cls(
                    true_positives=jnp.array(0, jnp.uint32),
                    true_negatives=jnp.array(0, jnp.uint32),
                    false_positives=jnp.array(0, jnp.uint32),
                    false_negatives=jnp.array(0, jnp.uint32),
                )

            @classmethod
            def from_model_output(cls, *, logits: jnp.array, labels: jnp.array, **_):
                preds = logits

                if from_logits:
                    preds = flax.linen.activation.sigmoid(preds)

                labels = labels > threshold
                preds = preds > threshold

                return cls(
                    true_positives=((preds == 1) & (labels == 1)).sum(),
                    true_negatives=((preds == 0) & (labels == 0)).sum(),
                    false_positives=((preds == 1) & (labels == 0)).sum(),
                    false_negatives=((preds == 0) & (labels == 1)).sum(),
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
                denominator = jax.lax.rsqrt(S * P * (1 - S) * (1 - P))
                return numerator * denominator

        return WithConfig
