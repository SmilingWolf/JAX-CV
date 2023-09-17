import flax
import jax.numpy as jnp
from clu import metrics


@flax.struct.dataclass
class F1Score(metrics.Metric):
    """
    Computes the micro-averaged F1 score
    from model outputs 'logits' and 'labels'.
    """

    @classmethod
    def with_config(cls, threshold: float, from_logits: bool):
        @flax.struct.dataclass
        class WithConfig(cls):
            true_positives: jnp.array
            false_positives: jnp.array
            false_negatives: jnp.array

            @classmethod
            def empty(cls):
                return cls(
                    true_positives=jnp.array(0, jnp.int32),
                    false_positives=jnp.array(0, jnp.int32),
                    false_negatives=jnp.array(0, jnp.int32),
                )

            @classmethod
            def from_model_output(cls, *, logits: jnp.array, labels: jnp.array, **_):
                preds = logits

                if from_logits:
                    preds = flax.linen.activation.sigmoid(preds)

                preds = preds > threshold
                return cls(
                    true_positives=((preds == 1) & (labels == 1)).sum(),
                    false_positives=((preds == 1) & (labels == 0)).sum(),
                    false_negatives=((preds == 0) & (labels == 1)).sum(),
                )

            def merge(self, other: metrics.Metric) -> metrics.Metric:
                return type(self)(
                    true_positives=self.true_positives + other.true_positives,
                    false_positives=self.false_positives + other.false_positives,
                    false_negatives=self.false_negatives + other.false_negatives,
                )

            def compute(self):
                numerator = 2 * self.true_positives
                denominator = (
                    (2 * self.true_positives)
                    + self.false_positives
                    + self.false_negatives
                )
                return numerator / denominator

        return WithConfig
