"""
LangGraph Agent 工作流编排。

节点执行顺序（含条件路由）：

  [START]
     │
     ▼
  intent_router          ← 识别意图 + 提取参数
     │
     ├─ intent=knowledge ──────────────────────┐
     │                                         ▼
     ▼                                    rag_retrieval
  data_diagnosis                               │
     │                                         ▼
     ▼                                   report_generator
  metric_attribution                           │
     │                                         ▼
     ├─ intent=knowledge → skip rec           [END]
     │
     ▼
  product_recommendation
     │
     ▼
  review_insight
     │
     ▼
  action_planner
     │
     ▼
  report_generator
     │
     ▼
  [END]

使用方式：
    from project1_app.agent_graph.graph import build_graph, run_graph

    graph = build_graph(text_to_sql=ts, rag_kb=rag)
    result = run_graph(graph, "最近 GMV 为什么下降？")
    print(result["final_report"])
"""
from __future__ import annotations

from functools import partial
from typing import Literal, Optional

from langgraph.graph import StateGraph, START, END

from project1_app.agent_graph.state import AgentState, empty_state
from project1_app.agent_graph.nodes import (
    node_intent_router,
    node_data_diagnosis,
    node_metric_attribution,
    node_product_recommendation,
    node_review_insight,
    node_rag_retrieval,
    node_action_planner,
    node_report_generator,
)


# ---------------------------------------------------------------------------
# 条件路由函数
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> Literal["data_diagnosis", "rag_retrieval"]:
    """知识问答类问题直接走 RAG，跳过数据查询和推荐链路。"""
    if state.get("intent") == "knowledge":
        return "rag_retrieval"
    return "data_diagnosis"


def route_after_attribution(state: AgentState) -> Literal["product_recommendation", "report_generator"]:
    """如果没有数据，跳过推荐直接生成报告。"""
    data = state.get("diagnosis_data", "")
    if not data or "失败" in data or "无数据" in data:
        return "report_generator"
    return "product_recommendation"


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def build_graph(
    text_to_sql=None,
    rag_kb=None,
    db_path: str = "",
) -> StateGraph:
    """
    构建 LangGraph 工作流。

    Args:
        text_to_sql: TextToSQL 实例（注入给 data_diagnosis 节点）
        rag_kb:      RAGKnowledgeBase 实例（注入给 rag_retrieval 节点）
        db_path:     SQLite 数据库路径（注入给 metric_attribution 节点）

    Returns:
        编译好的 CompiledGraph，调用 .invoke(state) 执行。
    """
    builder = StateGraph(AgentState)

    # 节点注册（使用 partial 注入依赖，保持节点函数签名纯净）
    builder.add_node("intent_router", node_intent_router)
    builder.add_node("data_diagnosis", partial(node_data_diagnosis, text_to_sql=text_to_sql))
    builder.add_node("metric_attribution", partial(node_metric_attribution, db_path=db_path))
    builder.add_node("product_recommendation", node_product_recommendation)
    builder.add_node("review_insight", node_review_insight)
    builder.add_node("rag_retrieval", partial(node_rag_retrieval, rag_kb=rag_kb))
    builder.add_node("action_planner", node_action_planner)
    builder.add_node("report_generator", node_report_generator)

    # 边：固定顺序
    builder.add_edge(START, "intent_router")

    # 边：条件路由 —— intent_router 之后分叉
    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "data_diagnosis": "data_diagnosis",
            "rag_retrieval": "rag_retrieval",
        },
    )

    # knowledge 路径：RAG → 报告
    builder.add_edge("rag_retrieval", "report_generator")

    # 诊断路径：数据 → 归因 → 条件路由（有数据则推荐，否则直接报告）
    builder.add_edge("data_diagnosis", "metric_attribution")
    builder.add_conditional_edges(
        "metric_attribution",
        route_after_attribution,
        {
            "product_recommendation": "product_recommendation",
            "report_generator": "report_generator",
        },
    )

    # 推荐路径：推荐 → 评论洞察 → 行动计划 → 报告
    builder.add_edge("product_recommendation", "review_insight")
    builder.add_edge("review_insight", "action_planner")
    builder.add_edge("action_planner", "report_generator")
    builder.add_edge("report_generator", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 便捷运行函数
# ---------------------------------------------------------------------------

def run_graph(graph, user_message: str) -> AgentState:
    """
    运行 LangGraph 工作流，返回完整最终状态。

    Args:
        graph:        build_graph() 返回的编译图
        user_message: 用户输入

    Returns:
        最终 AgentState（含 final_report 和 trajectory）
    """
    return graph.invoke(empty_state(user_message))


def stream_graph(graph, user_message: str):
    """
    流式运行，每个节点结束后 yield chunk。

    chunk 格式：{node_name: partial_state_update}
    Streamlit 用此函数实时展示节点执行进度。
    """
    for chunk in graph.stream(empty_state(user_message)):
        yield chunk


def format_trajectory(state: AgentState) -> str:
    """将执行轨迹格式化为可读日志，用于 Streamlit 展示。"""
    lines = ["**执行轨迹**"]
    for step in state.get("trajectory", []):
        lines.append(
            f"  [{step['node']}] {step['duration_ms']}ms — {step['summary']}"
        )
    errors = state.get("error_log", [])
    if errors:
        lines.append("\n**异常记录**")
        for err in errors:
            lines.append(f"  ⚠️ {err}")
    return "\n".join(lines)
