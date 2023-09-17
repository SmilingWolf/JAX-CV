import flax
import jax.numpy as jnp
from clu import metrics


@flax.struct.dataclass
class Recall(metrics.Metric):
    """Computes the recall from model outputs `logits` and `labels`."""

    true_positives: jnp.array
    false_negatives: jnp.array

    @classmethod
    def from_model_output(
        cls, *, threshold: float, logits: jnp.array, labels: jnp.array, **_
    ) -> metrics.Metric:
        preds = logits > threshold
        return cls(
            true_positives=((preds == 1) & (labels == 1)).sum(),
            false_negatives=((preds == 0) & (labels == 1)).sum(),
        )

    def merge(self, other: metrics.Metric) -> metrics.Metric:
        return type(self)(
            true_positives=self.true_positives + other.true_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_negatives)

@flax.struct.dataclass
class RecallV2(metrics.Metric):
    """Computes the recall from model outputs `logits` and `labels`."""

    @classmethod
    def with_threshold(cls, threshold:float):
    
        @flax.struct.dataclass
        class WithThreshold(cls):
            true_positives: jnp.array
            false_negatives: jnp.array

            @classmethod
            def from_model_output(cls, *, logits: jnp.array, labels: jnp.array, **_):
                preds = logits > threshold
                return cls(
                    true_positives=((preds == 1) & (labels == 1)).sum(),
                    false_negatives=((preds == 0) & (labels == 1)).sum(),
                )

            def merge(self, other: metrics.Metric) -> metrics.Metric:
                return type(self)(
                    true_positives=self.true_positives + other.true_positives,
                    false_negatives=self.false_negatives + other.false_negatives,
                )

            def compute(self):
                return self.true_positives / (self.true_positives + self.false_negatives)
        return WithThreshold