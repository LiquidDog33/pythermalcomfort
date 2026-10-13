"""Model output classifications shared with plotting, independent of presentation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _OutputClassification:
    """Ordered category upper edges, including the finite mapping limit."""

    upper_bounds: tuple[float, ...]
    labels: tuple[str, ...]
    right: bool

    @property
    def mapping(self) -> dict[float, str]:
        """Return a fresh mapping for the models' existing categorization helper."""
        return dict(zip(self.upper_bounds, self.labels, strict=True))


_PMV_BOUNDS = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 10.0)
_PMV_LABELS = (
    "Cold",
    "Cool",
    "Slightly Cool",
    "Neutral",
    "Slightly Warm",
    "Warm",
    "Hot",
)
PMV_ISO_CLASSIFICATION = _OutputClassification(
    upper_bounds=_PMV_BOUNDS, labels=_PMV_LABELS, right=False
)
PMV_ASHRAE_CLASSIFICATION = _OutputClassification(
    upper_bounds=_PMV_BOUNDS, labels=_PMV_LABELS, right=True
)
