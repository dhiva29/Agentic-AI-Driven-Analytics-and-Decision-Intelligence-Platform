# Agentic AI-Driven Analytics and Decision Intelligence Platform

Production-grade, schema-agnostic analytics application built with Streamlit, deterministic profiling, agentic orchestration (LangGraph), and FAISS-backed RAG.

## Features
- Upload any CSV and auto-detect schema (numeric, categorical, datetime, identifiers)
- Deterministic data understanding: quality, trends, distributions, correlations, outliers
- Agentic analysis workflow:
  1. Data Analyst Agent
  2. Visualization Planner Agent
  3. RAG Agent (FAISS)
  4. Dashboard Builder Agent
- Structured JSON dashboard schema and confidence scoring
- PowerBI-style responsive dashboard with KPI cards + Plotly visuals
- Agent insights tab with reasoning, citations, confidence, and audit trail
- Natural language chat mode for anomaly checks, root-cause hints, and forecasting
- SQLite audit logging (`agent_logs`) for decisions and overrides

## Folder Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
└── src
    ├── agents
    │   └── workflow.py
    ├── chat
    │   └── engine.py
    ├── config
    │   └── settings.py
    ├── core
    │   ├── data_analyzer.py
    │   └── models.py
    ├── dashboard
    │   └── renderer.py
    ├── db
    │   └── logger.py
    ├── rag
    │   └── faiss_retriever.py
    └── ui
        └── theme.py
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture Principles
- Deterministic analytics before LLM-like reasoning.
- Structured outputs in JSON-friendly dataclasses.
- LLM-style logic is advisory only; core decisions remain programmatic.
- Modular separation of concerns for production maintainability.
