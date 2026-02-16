from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnSchema:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    identifier: list[str] = field(default_factory=list)


@dataclass
class DataQualitySummary:
    row_count: int
    column_count: int
    missing_by_column: dict[str, float]
    duplicate_rows: int
    outlier_counts: dict[str, int]


@dataclass
class DatasetSummary:
    dataset_type: str
    potential_kpis: list[str]
    relationships_detected: list[str]
    time_series_available: bool
    recommended_analysis: list[str]
    trend_columns: list[str] = field(default_factory=list)
    distribution_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_type": self.dataset_type,
            "potential_kpis": self.potential_kpis,
            "relationships_detected": self.relationships_detected,
            "time_series_available": self.time_series_available,
            "recommended_analysis": self.recommended_analysis,
            "trend_columns": self.trend_columns,
            "distribution_notes": self.distribution_notes,
        }


@dataclass
class ChartSpec:
    type: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    title: str | None = None


@dataclass
class DashboardSchema:
    kpis: list[dict[str, Any]]
    charts: list[ChartSpec]
    insights: list[str]
    confidence_score: float
    rationale: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kpis": self.kpis,
            "charts": [c.__dict__ for c in self.charts],
            "insights": self.insights,
            "confidence_score": self.confidence_score,
            "rationale": self.rationale,
        }
