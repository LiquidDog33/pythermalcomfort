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
UTCI_CLASSIFICATION = _OutputClassification(
    upper_bounds=(-40.0, -27.0, -13.0, 0.0, 9.0, 26.0, 32.0, 38.0, 46.0, 1000.0),
    labels=(
        "extreme cold stress",
        "very strong cold stress",
        "strong cold stress",
        "moderate cold stress",
        "slight cold stress",
        "no thermal stress",
        "moderate heat stress",
        "strong heat stress",
        "very strong heat stress",
        "extreme heat stress",
    ),
    right=True,
)

HEAT_INDEX_CLASSIFICATION = _OutputClassification(
    upper_bounds=(27.0, 32.0, 41.0, 54.0, 1000.0),
    labels=("no risk", "caution", "extreme caution", "danger", "extreme danger"),
    right=True,
)
