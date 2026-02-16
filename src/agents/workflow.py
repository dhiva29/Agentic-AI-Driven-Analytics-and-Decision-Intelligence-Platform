from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = "__end__"
    StateGraph = None

from src.core.models import ChartSpec, ColumnSchema, DashboardSchema, DatasetSummary
from src.db.logger import AuditLogger
from src.rag.faiss_retriever import FAISSRetriever


class AgentState(TypedDict, total=False):
    df: pd.DataFrame
    schema: ColumnSchema
    summary: DatasetSummary
    analyst_output: dict
    planner_output: dict
    rag_output: dict
    dashboard_schema: DashboardSchema


class AgenticOrchestrator:
    def __init__(self) -> None:
        self.logger = AuditLogger()
        self.rag = FAISSRetriever()
        self.graph = self._build_graph()

    def run(self, df: pd.DataFrame, schema: ColumnSchema, summary: DatasetSummary) -> DashboardSchema:
        self.rag.build_knowledge_base(df)
        initial_state: AgentState = {"df": df, "schema": schema, "summary": summary}
        if self.graph is not None:
            result = self.graph.invoke(initial_state)
            return result["dashboard_schema"]

        state = self._data_analyst_agent(initial_state)
        state = self._visualization_planner_agent(state)
        state = self._rag_agent(state)
        state = self._dashboard_builder_agent(state)
        return state["dashboard_schema"]

    def _build_graph(self):
        if StateGraph is None:
            return None

        graph = StateGraph(AgentState)
        graph.add_node("data_analyst", self._data_analyst_agent)
        graph.add_node("viz_planner", self._visualization_planner_agent)
        graph.add_node("rag_agent", self._rag_agent)
        graph.add_node("dashboard_builder", self._dashboard_builder_agent)

        graph.set_entry_point("data_analyst")
        graph.add_edge("data_analyst", "viz_planner")
        graph.add_edge("viz_planner", "rag_agent")
        graph.add_edge("rag_agent", "dashboard_builder")
        graph.add_edge("dashboard_builder", END)

        return graph.compile()

    def _data_analyst_agent(self, state: AgentState) -> AgentState:
        summary = state["summary"]
        important_metrics = summary.potential_kpis[:6]
        selected_kpi_columns = [
            kpi.split()[1] for kpi in important_metrics if len(kpi.split()) > 1 and "(" in kpi
        ]

        output = {
            "important_metrics": important_metrics,
            "selected_kpis": selected_kpi_columns,
            "suggested_viz": ["line" if summary.time_series_available else "bar", "bar", "histogram"],
            "reasons": [
                "Selected highest-signal numeric metrics from deterministic profiling.",
                "Balanced trend, comparison, and distribution coverage.",
            ],
            "confidence": 0.84,
        }
        self.logger.log("data_analyst", summary.to_dict(), output, output["confidence"])
        state["analyst_output"] = output
        return state

    def _visualization_planner_agent(self, state: AgentState) -> AgentState:
        schema = state["schema"]
        summary = state["summary"]
        numeric = schema.numeric
        categorical = schema.categorical

        plans = []
        if summary.time_series_available and schema.datetime and numeric:
            plans.append({"type": "line", "x": schema.datetime[0], "y": numeric[0], "priority": 1})
        if categorical and numeric:
            plans.append({"type": "bar", "x": categorical[0], "y": numeric[0], "priority": 2})
        if numeric:
            plans.append({"type": "histogram", "x": numeric[min(1, len(numeric) - 1)], "priority": 3})
        if len(numeric) >= 2:
            plans.append({"type": "heatmap", "x": numeric[0], "y": numeric[1], "priority": 4})

        output = {
            "chart_plan": plans,
            "prioritization_reason": "Prioritized executive readability: trend, segment comparison, distribution, relationships.",
            "confidence": 0.82,
        }
        self.logger.log("visualization_planner", state["analyst_output"], output, output["confidence"])
        state["planner_output"] = output
        return state

    def _rag_agent(self, state: AgentState) -> AgentState:
        query = "What business context should guide KPI and dashboard interpretation?"
        result = self.rag.retrieve(query, top_k=4)
        output = {
            "query": query,
            "citations": result.snippets,
            "scores": result.scores,
            "confidence": float(np.mean(result.scores) if result.scores else 0.65),
        }
        self.logger.log("rag_agent", {"query": query}, output, output["confidence"])
        state["rag_output"] = output
        return state

    def _dashboard_builder_agent(self, state: AgentState) -> AgentState:
        df = state["df"]
        summary = state["summary"]
        plans = state["planner_output"]["chart_plan"]

        kpis = []
        for col in state["schema"].numeric[:4]:
            kpis.append(
                {
                    "name": col,
                    "value": float(df[col].sum(skipna=True)),
                    "mean": float(df[col].mean(skipna=True)),
                    "delta": float(df[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).mean()),
                }
            )

        charts = [
            ChartSpec(type=p["type"], x=p.get("x"), y=p.get("y"), title=f"{p['type'].title()} View")
            for p in plans
        ]

        confidence = round(
            float(
                np.mean(
                    [
                        state["analyst_output"]["confidence"],
                        state["planner_output"]["confidence"],
                        state["rag_output"]["confidence"],
                    ]
                )
            ),
            2,
        )

        dashboard = DashboardSchema(
            kpis=kpis,
            charts=charts,
            insights=summary.relationships_detected[:5] or summary.recommended_analysis[:5],
            confidence_score=confidence,
            rationale={
                "kpi_selection": state["analyst_output"]["reasons"],
                "visualization_choice": [state["planner_output"]["prioritization_reason"]],
                "rag_grounding": state["rag_output"]["citations"],
            },
        )

        self.logger.log(
            "dashboard_builder",
            {"summary": summary.to_dict(), "chart_plan": plans},
            dashboard.to_dict(),
            dashboard.confidence_score,
        )

        state["dashboard_schema"] = dashboard
        return state
