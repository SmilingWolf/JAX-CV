import flax
from clu import metrics


@flax.struct.dataclass
class Precision(metrics.Metric):
    """Computes the precision from model outputs `logits` and `labels`."""

    threshold: float
    true_positives: jnp.array
    pred_positives: jnp.array

    @classmethod
    def from_model_output(
        cls, *, logits: jnp.array, labels: jnp.array, **_
    ) -> metrics.Metric:
        preds = logits > self.threshold
        return cls(
            true_positives=((preds == 1) & (labels == 1)).sum(),
            pred_positives=(preds == 1).sum(),
        )

    def merge(self, other: metrics.Metric) -> metrics.Metric:
        # Note that for precision we cannot average metric values because the
        # denominator of the metric value is pred_positives and not every batch of
        # examples has the same number of pred_positives (as opposed to e.g.
        # accuracy where every batch has the same number of)
        return type(self)(
            threshold=self.threshold,
            true_positives=self.true_positives + other.true_positives,
            pred_positives=self.pred_positives + other.pred_positives,
        )

    def compute(self):
        return self.true_positives / self.pred_positives
