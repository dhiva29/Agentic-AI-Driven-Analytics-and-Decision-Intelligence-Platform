from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.agents.workflow import AgenticOrchestrator
from src.chat.engine import ChatEngine
from src.config.settings import SETTINGS
from src.core.data_analyzer import DataAnalyzer
from src.dashboard.renderer import DashboardRenderer
from src.db.logger import AuditLogger
from src.ui.theme import apply_theme

st.set_page_config(page_title=SETTINGS.project_name, layout="wide")

if "analyzer" not in st.session_state:
    st.session_state.analyzer = DataAnalyzer()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgenticOrchestrator()
if "logger" not in st.session_state:
    st.session_state.logger = AuditLogger()

analyzer: DataAnalyzer = st.session_state.analyzer
orchestrator: AgenticOrchestrator = st.session_state.orchestrator

with st.sidebar:
    st.title("⚙️ Control Center")
    dark_mode = st.toggle("Dark Theme", value=True)
    user_mode = st.radio("Build Mode", ["Review Suggested Dashboard", "Auto Build Dashboard"])

apply_theme(dark_mode)

st.title("Agentic AI-Driven Analytics and Decision Intelligence Platform")
st.caption("Production-grade, schema-agnostic AI analytics with deterministic intelligence + agentic reasoning")

pages = ["Data Source", "Dashboard", "Agent Insights", "Chat"]
page = st.segmented_control("Navigation", pages, default="Data Source")

if page == "Data Source":
    st.subheader("Upload and Understand Your Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        valid, msg = analyzer.validate_csv(uploaded)
        if not valid:
            st.error(msg)
        else:
            st.success(msg)
            df = analyzer.load_dataframe(uploaded)
            schema = analyzer.detect_schema(df)
            quality = analyzer.data_quality(df, schema)
            summary = analyzer.summarize_dataset(df, schema)

            st.session_state.df = df
            st.session_state.schema = schema
            st.session_state.summary = summary

            st.markdown("### Preview")
            st.dataframe(df.head(SETTINGS.max_preview_rows), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Detected Schema")
                st.json(schema.__dict__)
            with c2:
                st.markdown("### Data Quality Summary")
                st.json(quality.__dict__)

            st.markdown("### Structured Summary")
            st.code(json.dumps(summary.to_dict(), indent=2), language="json")

if page == "Dashboard":
    st.subheader("PowerBI-Style Decision Dashboard")
    if "df" not in st.session_state:
        st.info("Upload a CSV from Data Source tab to continue.")
    else:
        if st.button("Run Agentic Analysis", type="primary"):
            dashboard_schema = orchestrator.run(
                st.session_state.df,
                st.session_state.schema,
                st.session_state.summary,
            )
            st.session_state.dashboard_schema = dashboard_schema

        if "dashboard_schema" in st.session_state:
            dashboard_schema = st.session_state.dashboard_schema
            if user_mode == "Review Suggested Dashboard":
                st.markdown("### Suggested Dashboard Rationale")
                st.json(dashboard_schema.rationale)
                accepted = st.checkbox("Accept this dashboard plan", value=True)
                if accepted:
                    DashboardRenderer().render(st.session_state.df, dashboard_schema)
                else:
                    st.warning("Modify settings and rerun analysis.")
            else:
                DashboardRenderer().render(st.session_state.df, dashboard_schema)
        else:
            st.info("Click 'Run Agentic Analysis' to generate dashboard schema.")

if page == "Agent Insights":
    st.subheader("Agent Reasoning & Grounded Intelligence")
    logs = st.session_state.logger.fetch_recent(limit=100)
    if not logs:
        st.info("No logs yet. Run analysis first.")
    else:
        for record in logs:
            with st.expander(f"{record['agent_type']} @ {record['timestamp']}"):
                st.write("**Confidence:**", f"{record['confidence']:.2f}")
                st.write("**Input**")
                st.json(record["input"])
                st.write("**Output**")
                st.json(record["output"])

if page == "Chat":
    st.subheader("Natural Language Analytics Chat")
    if "df" not in st.session_state:
        st.info("Upload data and run analysis to enable chat intelligence.")
    else:
        chat = ChatEngine(orchestrator.rag)
        query = st.text_input("Ask a question", placeholder="Why did revenue drop? / Forecast next month / What anomalies exist?")
        if st.button("Ask") and query.strip():
            result = chat.answer(query, st.session_state.df, st.session_state.get("dashboard_schema"))
            st.write(result["text"])
            if result["chart"] is not None:
                st.plotly_chart(result["chart"], use_container_width=True)
            st.caption(f"Recommendation: {result['recommendation']}")
            st.caption(f"Confidence: {result['confidence']:.2f}")
            if result["citations"]:
                st.markdown("**RAG Citations**")
                for cite in result["citations"]:
                    st.caption(f"- {cite}")
