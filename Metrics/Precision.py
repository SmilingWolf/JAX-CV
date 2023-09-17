import flax
import jax.numpy as jnp
from clu import metrics


@flax.struct.dataclass
class Precision(metrics.Metric):
    """
    Computes the micro-averaged precision
    from model outputs 'logits' and 'labels'.
    """

    @classmethod
    def with_config(cls, threshold: float, from_logits: bool):
        @flax.struct.dataclass
        class WithConfig(cls):
            true_positives: jnp.array
            pred_positives: jnp.array

            @classmethod
            def empty(cls):
                return cls(
                    true_positives=jnp.array(0, jnp.int32),
                    pred_positives=jnp.array(0, jnp.int32),
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
                    pred_positives=(preds == 1).sum(),
                )

            def merge(self, other: metrics.Metric) -> metrics.Metric:
                return type(self)(
                    true_positives=self.true_positives + other.true_positives,
                    pred_positives=self.pred_positives + other.pred_positives,
                )

            def compute(self):
                return self.true_positives / self.pred_positives

        return WithConfig
