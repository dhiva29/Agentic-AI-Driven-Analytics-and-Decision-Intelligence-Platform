from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.models import DashboardSchema


class DashboardRenderer:
    def render(self, df: pd.DataFrame, schema: DashboardSchema) -> None:
        self._render_kpis(schema)
        self._render_charts(df, schema)
        self._render_bottom(schema)

    def _render_kpis(self, schema: DashboardSchema) -> None:
        st.subheader("Executive KPIs")
        cols = st.columns(max(len(schema.kpis), 1))
        for i, kpi in enumerate(schema.kpis):
            with cols[i % len(cols)]:
                st.metric(
                    label=kpi["name"],
                    value=f"{kpi['value']:,.2f}",
                    delta=f"{kpi['delta'] * 100:.2f}% avg change",
                )

    def _render_charts(self, df: pd.DataFrame, schema: DashboardSchema) -> None:
        st.subheader("Dashboard Visuals")
        left, right = st.columns(2)

        for idx, chart in enumerate(schema.charts):
            container = left if idx % 2 == 0 else right
            with container:
                fig = self._build_figure(df, chart.__dict__)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

    def _build_figure(self, df: pd.DataFrame, spec: dict):
        chart_type = spec["type"]
        x = spec.get("x")
        y = spec.get("y")
        title = spec.get("title", chart_type.title())

        if chart_type == "line" and x in df.columns and y in df.columns:
            return px.line(df, x=x, y=y, title=title, template="plotly_dark")
        if chart_type == "bar" and x in df.columns and y in df.columns:
            grouped = df.groupby(x, as_index=False)[y].mean().head(20)
            return px.bar(grouped, x=x, y=y, title=title, template="plotly_dark")
        if chart_type == "histogram" and x in df.columns:
            return px.histogram(df, x=x, title=title, template="plotly_dark")
        if chart_type == "heatmap" and x in df.columns and y in df.columns:
            corr = df[[x, y]].corr(numeric_only=True)
            return px.imshow(corr, text_auto=True, title=title, template="plotly_dark")
        return None

    def _render_bottom(self, schema: DashboardSchema) -> None:
        st.subheader("Insights & Recommendations")
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.markdown("#### Insight Panel")
            for item in schema.insights:
                st.write(f"• {item}")
        with c2:
            st.markdown("#### Agent Recommendation Panel")
            for section, notes in schema.rationale.items():
                st.markdown(f"**{section.replace('_', ' ').title()}**")
                for note in notes[:3]:
                    st.caption(f"- {note}")
        with c3:
            st.markdown("#### Confidence")
            st.metric("Score", f"{schema.confidence_score:.2f}")
