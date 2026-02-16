from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.core.models import ColumnSchema, DataQualitySummary, DatasetSummary


class DataAnalyzer:
    """Deterministic analytics engine for schema-agnostic dataset understanding."""

    def validate_csv(self, file) -> tuple[bool, str]:
        if file is None:
            return False, "No file uploaded"
        if not file.name.lower().endswith(".csv"):
            return False, "Only CSV files are supported"
        return True, "Valid file"

    def load_dataframe(self, file) -> pd.DataFrame:
        return pd.read_csv(file)

    def detect_schema(self, df: pd.DataFrame) -> ColumnSchema:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols: list[str] = []
        categorical_cols: list[str] = []
        identifier_cols: list[str] = []

        for col in df.columns:
            series = df[col]
            if col in numeric_cols:
                continue

            looks_temporal = series.astype(str).str.contains(r"[-/:]|\d{4}", regex=True, na=False).mean() > 0.6
            if looks_temporal:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    datetime_cols.append(col)
                    continue

            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
            if unique_ratio > 0.95:
                identifier_cols.append(col)
            else:
                categorical_cols.append(col)

        for col in numeric_cols:
            unique_ratio = df[col].nunique(dropna=True) / max(len(df[col]), 1)
            is_integer_like = pd.api.types.is_integer_dtype(df[col])
            monotonic = df[col].dropna().is_monotonic_increasing
            if unique_ratio > 0.99 and is_integer_like and monotonic:
                identifier_cols.append(col)

        return ColumnSchema(
            numeric=numeric_cols,
            categorical=categorical_cols,
            datetime=datetime_cols,
            identifier=identifier_cols,
        )

    def data_quality(self, df: pd.DataFrame, schema: ColumnSchema) -> DataQualitySummary:
        missing_by_column = ((df.isna().sum() / len(df)) * 100).round(2).to_dict()
        duplicate_rows = int(df.duplicated().sum())
        outlier_counts = self._detect_outliers(df, schema.numeric)

        return DataQualitySummary(
            row_count=len(df),
            column_count=df.shape[1],
            missing_by_column=missing_by_column,
            duplicate_rows=duplicate_rows,
            outlier_counts=outlier_counts,
        )

    def summarize_dataset(self, df: pd.DataFrame, schema: ColumnSchema) -> DatasetSummary:
        corr_rel = self._find_correlations(df, schema.numeric)
        trend_columns = self._trend_columns(df, schema)
        kpis = self._potential_kpis(df, schema)
        dataset_type = self._dataset_type(schema)

        recommended_analysis = [
            "KPI Monitoring",
            "Segmentation by categorical dimensions",
            "Outlier root-cause review",
        ]
        if schema.datetime:
            recommended_analysis.append("Time-series trend and seasonality")

        distributions = [
            f"{col}: skew={df[col].skew(skipna=True):.2f}"
            for col in schema.numeric[:5]
            if not df[col].dropna().empty
        ]

        return DatasetSummary(
            dataset_type=dataset_type,
            potential_kpis=kpis,
            relationships_detected=corr_rel,
            time_series_available=bool(schema.datetime),
            recommended_analysis=recommended_analysis,
            trend_columns=trend_columns,
            distribution_notes=distributions,
        )

    def as_pretty_json(self, obj) -> str:
        return json.dumps(asdict(obj), indent=2, default=str)

    def _find_correlations(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
        if len(numeric_cols) < 2:
            return []
        corr = df[numeric_cols].corr(numeric_only=True)
        findings: list[str] = []

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                val = corr.loc[col_a, col_b]
                if pd.notna(val) and abs(val) >= 0.6:
                    findings.append(f"{col_a} vs {col_b}: correlation {val:.2f}")

        return findings[:10]

    def _trend_columns(self, df: pd.DataFrame, schema: ColumnSchema) -> list[str]:
        if not schema.datetime or not schema.numeric:
            return []

        trends = []
        for num_col in schema.numeric[:10]:
            series = df[num_col].dropna().reset_index(drop=True)
            if len(series) < 5:
                continue
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            if abs(slope) > np.std(series) * 0.01:
                direction = "upward" if slope > 0 else "downward"
                trends.append(f"{num_col}: {direction} trend")
        return trends

    def _potential_kpis(self, df: pd.DataFrame, schema: ColumnSchema) -> list[str]:
        kpis = []
        for col in schema.numeric[:8]:
            total = float(df[col].sum(skipna=True))
            mean = float(df[col].mean(skipna=True))
            kpis.append(f"Total {col} ({total:,.2f})")
            kpis.append(f"Avg {col} ({mean:,.2f})")
        return kpis[:10]

    def _dataset_type(self, schema: ColumnSchema) -> str:
        if schema.datetime and schema.numeric:
            return "Time Series + Tabular"
        if schema.numeric and schema.categorical:
            return "Analytical Tabular"
        return "Generic Structured"

    def _detect_outliers(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, int]:
        outliers: dict[str, int] = {}
        for col in numeric_cols[:10]:
            series = df[[col]].dropna()
            if len(series) < 20:
                outliers[col] = 0
                continue
            model = IsolationForest(contamination=0.05, random_state=42)
            labels = model.fit_predict(series)
            outliers[col] = int((labels == -1).sum())
        return outliers
