"""Reference-capacity diagnostics: parallel aliases cannot earn repeated credit."""

from __future__ import annotations

from dataclasses import dataclass

from .edge_quality import EdgeTypeQuality


@dataclass(frozen=True, slots=True)
class ReferenceCapacity:
    precision: float | None
    recall: float | None
    credited_reference_length_m: float
    raw_predicted_length_m: float
    uncredited_length_m: float
    matched_prediction_surplus_m: float


def reference_capacity(quality: EdgeTypeQuality) -> ReferenceCapacity:
    """Credit covered reference length once, even for noncoincident nearby aliases.

    This is a conservative projected-length diagnostic for exhaustive same-class
    references, not a physical-rim precision certificate for incomplete labels.
    The surplus also includes small angular projection differences.
    """
    raw = quality.raw_predicted_length
    credit = quality.matched_reference_length
    return ReferenceCapacity(
        credit / raw if raw else None,
        quality.recall,
        credit,
        raw,
        max(0.0, raw - credit),
        max(0.0, quality.matched_predicted_length - credit),
    )
