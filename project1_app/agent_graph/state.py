"""
LangGraph Agent 共享状态。

所有节点读写同一个 AgentState 对象，
LangGraph 会在节点间自动传递并合并（Annotated + operator.add 处理列表追加）。
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ---- 输入 ----
    user_message: str                        # 用户原始问题

    # ---- 意图识别结果 ----
    intent: str                              # "diagnosis" / "recommendation" / "content" / "knowledge" / "general"
    problem_type: str                        # "CVR下降" / "CTR下降" 等
    business_goal: str                       # "提升GMV" / "提升CVR" 等
    target_category: str                     # 目标品类，可为空
    price_range: list[float]                 # [min, max]

    # ---- 经营数据诊断结果 ----
    diagnosis_sql: str                       # 生成的 SQL
    diagnosis_data: str                      # 查询结果（字符串化）
    diagnosis_summary: str                   # LLM 归因摘要

    # ---- 商品推荐结果 ----
    recommended_skus: Annotated[list[dict], operator.add]   # Top-K SKU 列表（支持追加）
    recommendation_text: str                 # 格式化推荐文本

    # ---- 评论洞察结果 ----
    review_insights: Annotated[list[dict], operator.add]    # 各 SKU 洞察（支持追加）
    review_insight_text: str                 # 格式化洞察文本

    # ---- RAG 知识库结果 ----
    rag_answer: str
    rag_sources: list[str]

    # ---- 行动计划 ----
    action_tasks: list[dict]                 # 结构化任务列表
    action_plan_text: str                    # 格式化任务文本

    # ---- 最终输出 ----
    final_report: str                        # 汇总报告，供前端展示

    # ---- 执行轨迹（可观测性）----
    trajectory: Annotated[list[dict], operator.add]         # 每个节点的执行记录
    error_log: Annotated[list[str], operator.add]           # 节点级错误，不中断流程


def empty_state(user_message: str) -> AgentState:
    """创建一个干净的初始状态。"""
    return AgentState(
        user_message=user_message,
        intent="general",
        problem_type="",
        business_goal="提升GMV",
        target_category="",
        price_range=[0.0, 9999.0],
        diagnosis_sql="",
        diagnosis_data="",
        diagnosis_summary="",
        recommended_skus=[],
        recommendation_text="",
        review_insights=[],
        review_insight_text="",
        rag_answer="",
        rag_sources=[],
        action_tasks=[],
        action_plan_text="",
        final_report="",
        trajectory=[],
        error_log=[],
    )
