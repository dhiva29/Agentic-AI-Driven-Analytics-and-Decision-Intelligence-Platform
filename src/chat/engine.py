from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from src.core.models import DashboardSchema
from src.rag.faiss_retriever import FAISSRetriever


class ChatEngine:
    def __init__(self, retriever: FAISSRetriever) -> None:
        self.retriever = retriever

    def answer(self, query: str, df: pd.DataFrame, dashboard: DashboardSchema | None) -> dict:
        lower_q = query.lower()
        context = self.retriever.retrieve(query, top_k=3)

        if "forecast" in lower_q:
            return self._forecast(df, context.snippets)
        if "anomal" in lower_q:
            return self._anomalies(df, context.snippets)
        if "drop" in lower_q or "why" in lower_q:
            return self._reason(df, dashboard, context.snippets)

        return {
            "text": "I analyzed the dataset context and suggest checking KPI trends and outlier segments.",
            "chart": None,
            "recommendation": "Use Dashboard tab to compare trend + category performance.",
            "citations": context.snippets,
            "confidence": 0.73,
        }

    def _forecast(self, df: pd.DataFrame, citations: list[str]) -> dict:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            return {"text": "No numeric series available for forecasting.", "chart": None, "recommendation": "Upload richer data.", "citations": citations, "confidence": 0.4}

        col = numeric[0]
        series = df[col].dropna().tail(12).reset_index(drop=True)
        if len(series) < 4:
            return {"text": "Insufficient history to forecast.", "chart": None, "recommendation": "Need at least 4 points.", "citations": citations, "confidence": 0.45}

        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        next_val = float(intercept + slope * (len(series)))
        fig = px.line(x=list(range(len(series))) + [len(series)], y=series.tolist() + [next_val], title=f"Forecast for {col}")
        return {
            "text": f"Projected next value for {col} is {next_val:,.2f} based on linear trend.",
            "chart": fig,
            "recommendation": "Validate with domain seasonality before action.",
            "citations": citations,
            "confidence": 0.68,
        }

    def _anomalies(self, df: pd.DataFrame, citations: list[str]) -> dict:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            return {"text": "No numeric fields for anomaly detection.", "chart": None, "recommendation": "Upload numeric metrics.", "citations": citations, "confidence": 0.4}

        col = numeric[0]
        s = df[col].dropna()
        z = (s - s.mean()) / (s.std() or 1)
        outliers = s[abs(z) > 3]
        fig = px.scatter(x=s.index, y=s.values, title=f"Anomaly scan: {col}")
        if not outliers.empty:
            fig.add_scatter(x=outliers.index, y=outliers.values, mode="markers", marker=dict(color="red", size=10), name="Outliers")
        return {
            "text": f"Detected {len(outliers)} high-z-score anomalies in {col}.",
            "chart": fig,
            "recommendation": "Investigate source events around highlighted points.",
            "citations": citations,
            "confidence": 0.75,
        }

    def _reason(self, df: pd.DataFrame, dashboard: DashboardSchema | None, citations: list[str]) -> dict:
        insights = dashboard.insights if dashboard else ["No dashboard insights yet."]
        return {
            "text": "Possible drivers include recent trend reversals, correlated metric declines, or segment concentration.",
            "chart": None,
            "recommendation": f"Start with: {insights[0] if insights else 'Run full analysis'}",
            "citations": citations,
            "confidence": 0.72,
        }
