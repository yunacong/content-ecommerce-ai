"""
LangGraph Agent 节点集合。

每个节点是一个纯函数：AgentState → dict（返回要更新的字段）。
节点内部捕获异常并写入 error_log，不向上抛出，保证图的连续执行。

节点列表：
  intent_router          意图识别 + 参数提取
  data_diagnosis         Text-to-SQL 经营数据查询
  metric_attribution     指标归因树（GMV 分解）
  product_recommendation 多模态选品推荐
  review_insight         评论洞察（Top-1 SKU）
  rag_retrieval          RAG 运营知识库
  action_planner         行动计划生成
  report_generator       最终报告汇总
"""
from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from project1_app.agent_graph.state import AgentState
from project1_app.tools.merchant_rec_tool import recommend_products
from project1_app.tools.review_insight_tool import get_review_insight
from project1_app.action_plan.task_generator import generate_action_plan, format_action_plan_text


def _llm_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _record(node: str, duration: float, summary: str) -> dict:
    return {"node": node, "duration_ms": round(duration * 1000), "summary": summary}


# ---------------------------------------------------------------------------
# 节点 1：意图识别
# ---------------------------------------------------------------------------
_INTENT_PROMPT = """你是电商运营 AI。分析用户问题，提取意图和关键参数。

用户问题：{message}

输出严格 JSON（不要多余文字）：
{{
  "intent": "diagnosis|recommendation|content|knowledge|general",
  "problem_type": "CVR下降|CTR下降|曝光下降|客单价下降|综合指标下滑|",
  "business_goal": "提升GMV|提升CTR|提升CVR|清库存|新品冷启动|提升GMV",
  "target_category": "品类名或空字符串",
  "price_range": [最低价, 最高价]
}}

intent 说明：
- diagnosis: 问经营数据/问题原因
- recommendation: 问推什么品/选品
- content: 问内容诊断/标题封面
- knowledge: 问平台规则/运营方法
- general: 复合问题或无法归类（按 diagnosis+recommendation 处理）"""


def node_intent_router(state: AgentState) -> dict:
    t0 = time.time()
    client = _llm_client()
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": _INTENT_PROMPT.format(message=state["user_message"])}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end]) if start != -1 else {}

        result = {
            "intent": parsed.get("intent", "general"),
            "problem_type": parsed.get("problem_type", "综合指标下滑"),
            "business_goal": parsed.get("business_goal", "提升GMV"),
            "target_category": parsed.get("target_category", ""),
            "price_range": parsed.get("price_range", [0.0, 9999.0]),
            "trajectory": [_record("intent_router", time.time() - t0,
                                   f"意图={parsed.get('intent')} 问题={parsed.get('problem_type')}")],
        }
    except Exception as e:
        logger.warning(f"intent_router 失败，使用默认值: {e}")
        result = {
            "intent": "general",
            "problem_type": "综合指标下滑",
            "business_goal": "提升GMV",
            "target_category": "",
            "price_range": [0.0, 9999.0],
            "trajectory": [_record("intent_router", time.time() - t0, "LLM失败，使用默认意图")],
            "error_log": [f"intent_router: {e}"],
        }
    return result


# ---------------------------------------------------------------------------
# 节点 2：经营数据诊断（Text-to-SQL）
# ---------------------------------------------------------------------------
def node_data_diagnosis(state: AgentState, text_to_sql=None) -> dict:
    """需要注入 text_to_sql 实例，通过 graph.py 中的 partial 绑定。"""
    t0 = time.time()
    if text_to_sql is None:
        return {
            "diagnosis_data": "数据查询模块未初始化",
            "trajectory": [_record("data_diagnosis", 0, "跳过（模块未注入）")],
        }
    try:
        question = state["user_message"]
        sql = text_to_sql.generate_sql(question)
        df, error = text_to_sql.execute_sql(sql)
        if error:
            data_str = f"查询出错: {error}"
        else:
            data_str = df.to_string(index=False, max_rows=20) if not df.empty else "无数据"

        return {
            "diagnosis_sql": sql,
            "diagnosis_data": data_str,
            "trajectory": [_record("data_diagnosis", time.time() - t0,
                                   f"SQL执行完成，返回 {0 if error else len(df)} 行")],
        }
    except Exception as e:
        return {
            "diagnosis_data": f"数据查询失败: {e}",
            "trajectory": [_record("data_diagnosis", time.time() - t0, f"异常: {e}")],
            "error_log": [f"data_diagnosis: {e}"],
        }


# ---------------------------------------------------------------------------
# 节点 3：指标归因
# ---------------------------------------------------------------------------
def node_metric_attribution(state: AgentState, db_path: str = "") -> dict:
    """基于查询数据，用 LLM 做 GMV 归因总结。"""
    t0 = time.time()
    data = state.get("diagnosis_data", "")
    if not data or "无数据" in data or "失败" in data:
        return {
            "diagnosis_summary": "数据不足，无法完成指标归因。",
            "trajectory": [_record("metric_attribution", time.time() - t0, "跳过（无数据）")],
        }

    # 优先用 metric_attribution 模块做结构化归因
    if db_path:
        try:
            from project1_app.diagnosis.metric_attribution import analyze_gmv_drop, attribution_to_dict
            import datetime
            today = datetime.date.today()
            cur_end = today.isoformat()
            cur_start = (today - datetime.timedelta(days=7)).isoformat()
            prev_end = (today - datetime.timedelta(days=8)).isoformat()
            prev_start = (today - datetime.timedelta(days=15)).isoformat()

            result = analyze_gmv_drop(db_path, cur_start, cur_end, prev_start, prev_end,
                                      state.get("target_category", ""))
            summary = "\n".join(result.evidence)

            return {
                "diagnosis_summary": summary,
                "problem_type": result.main_problem or state.get("problem_type", ""),
                "business_goal": result.recommendation_request.get("business_goal", state.get("business_goal", "")),
                "target_category": result.affected_category or state.get("target_category", ""),
                "trajectory": [_record("metric_attribution", time.time() - t0,
                                       f"归因完成: {result.main_problem}")],
            }
        except Exception as e:
            logger.warning(f"metric_attribution 模块失败，降级 LLM: {e}")

    # 降级：LLM 归因总结
    client = _llm_client()
    try:
        prompt = f"""根据以下电商经营数据，用 2-3 句话指出 GMV 变化的主因（曝光/CTR/CVR/AOV 哪个下降最多，哪个 SKU 拖累最大）。
数据：
{data[:1500]}
输出要简洁，包含具体数字。"""
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=300,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"数据已获取，归因分析失败: {e}"

    return {
        "diagnosis_summary": summary,
        "trajectory": [_record("metric_attribution", time.time() - t0, "LLM归因完成")],
    }


# ---------------------------------------------------------------------------
# 节点 4：商品推荐
# ---------------------------------------------------------------------------
def node_product_recommendation(state: AgentState) -> dict:
    t0 = time.time()
    params = {
        "business_goal": state.get("business_goal", "提升GMV"),
        "problem_type": state.get("problem_type", "综合指标下滑"),
        "target_category": state.get("target_category", ""),
        "price_range": state.get("price_range", [0, 9999]),
        "top_k": 5,
    }
    try:
        rec_text = recommend_products(params)
        # 简单解析推荐文本，提取 SKU 列表（用于后续评论洞察）
        skus = _parse_skus_from_text(rec_text)
        return {
            "recommendation_text": rec_text,
            "recommended_skus": skus,
            "trajectory": [_record("product_recommendation", time.time() - t0,
                                   f"推荐 {len(skus)} 个 SKU")],
        }
    except Exception as e:
        return {
            "recommendation_text": f"推荐服务异常: {e}",
            "trajectory": [_record("product_recommendation", time.time() - t0, f"异常: {e}")],
            "error_log": [f"product_recommendation: {e}"],
        }


def _parse_skus_from_text(text: str) -> list[dict]:
    """从格式化推荐文本中简单提取 SKU 信息（用于传给评论洞察节点）。"""
    import re
    skus = []
    # 匹配 "SKU_xxx" 或前缀 "parent_asin" 等
    ids = re.findall(r"(SKU_\w+|[A-Z0-9]{10})", text)
    for sid in ids[:5]:
        skus.append({"sku_id": sid, "product_name": sid})
    return skus


# ---------------------------------------------------------------------------
# 节点 5：评论洞察（对 Top-1 推荐 SKU）
# ---------------------------------------------------------------------------
def node_review_insight(state: AgentState) -> dict:
    t0 = time.time()
    skus = state.get("recommended_skus", [])
    if not skus:
        return {
            "review_insight_text": "无推荐 SKU，跳过评论洞察。",
            "trajectory": [_record("review_insight", 0, "跳过（无推荐结果）")],
        }

    # 只对 Top-1 做洞察（节省时间）
    top_sku = skus[0]
    sku_id = top_sku.get("sku_id", "")
    try:
        insight_text = get_review_insight({"sku_id": sku_id, "use_llm": True})
        return {
            "review_insights": [{"sku_id": sku_id, "text": insight_text}],
            "review_insight_text": insight_text,
            "trajectory": [_record("review_insight", time.time() - t0,
                                   f"{sku_id} 洞察完成")],
        }
    except Exception as e:
        return {
            "review_insight_text": f"评论洞察失败: {e}",
            "trajectory": [_record("review_insight", time.time() - t0, f"异常: {e}")],
            "error_log": [f"review_insight: {e}"],
        }


# ---------------------------------------------------------------------------
# 节点 6：RAG 知识库检索
# ---------------------------------------------------------------------------
def node_rag_retrieval(state: AgentState, rag_kb=None) -> dict:
    t0 = time.time()
    if rag_kb is None:
        return {
            "rag_answer": "",
            "trajectory": [_record("rag_retrieval", 0, "跳过（模块未注入）")],
        }
    intent = state.get("intent", "general")
    # 只有 knowledge 类问题才做 RAG，其他类型跳过避免噪音
    if intent not in ("knowledge", "general"):
        return {
            "rag_answer": "",
            "trajectory": [_record("rag_retrieval", 0, f"跳过（intent={intent}）")],
        }
    try:
        result = rag_kb.answer(state["user_message"])
        return {
            "rag_answer": result.get("answer", ""),
            "rag_sources": result.get("sources", []),
            "trajectory": [_record("rag_retrieval", time.time() - t0, "RAG检索完成")],
        }
    except Exception as e:
        return {
            "rag_answer": "",
            "trajectory": [_record("rag_retrieval", time.time() - t0, f"异常: {e}")],
            "error_log": [f"rag_retrieval: {e}"],
        }


# ---------------------------------------------------------------------------
# 节点 7：行动计划
# ---------------------------------------------------------------------------
def node_action_planner(state: AgentState) -> dict:
    t0 = time.time()
    skus = state.get("recommended_skus", [])
    if not skus:
        return {
            "action_plan_text": "无推荐商品，无法生成行动计划。",
            "trajectory": [_record("action_planner", 0, "跳过（无推荐 SKU）")],
        }
    try:
        diagnosis = {
            "problem_type": state.get("problem_type", "综合指标下滑"),
            "affected_category": state.get("target_category", ""),
        }
        client = _llm_client()
        tasks = generate_action_plan(
            diagnosis=diagnosis,
            recommended_skus=skus,
            use_llm=True,
            llm_client=client,
        )
        plan_text = format_action_plan_text(tasks)
        return {
            "action_tasks": tasks,
            "action_plan_text": plan_text,
            "trajectory": [_record("action_planner", time.time() - t0,
                                   f"生成 {len(tasks)} 条任务")],
        }
    except Exception as e:
        return {
            "action_plan_text": f"行动计划生成失败: {e}",
            "trajectory": [_record("action_planner", time.time() - t0, f"异常: {e}")],
            "error_log": [f"action_planner: {e}"],
        }


# ---------------------------------------------------------------------------
# 节点 8：最终报告生成
# ---------------------------------------------------------------------------
_REPORT_PROMPT = """你是内容电商 AI 经营助手。根据以下各模块分析结果，为商家生成一份简洁的经营建议报告。

用户问题：{user_message}

【经营数据归因】
{diagnosis_summary}

【商品推荐结果】
{recommendation_text}

【评论洞察】
{review_insight_text}

【知识库补充】
{rag_answer}

【行动计划】
{action_plan_text}

要求：
- 不超过400字
- 分「问题诊断」「推荐商品」「内容方向」「立即行动」四个小节
- 语言直接，不用客套话
- 如果某个模块结果为空，跳过该部分"""


def node_report_generator(state: AgentState) -> dict:
    t0 = time.time()
    client = _llm_client()
    try:
        prompt = _REPORT_PROMPT.format(
            user_message=state.get("user_message", ""),
            diagnosis_summary=state.get("diagnosis_summary", "无") or "无",
            recommendation_text=state.get("recommendation_text", "无") or "无",
            review_insight_text=state.get("review_insight_text", "无") or "无",
            rag_answer=state.get("rag_answer", "") or "无",
            action_plan_text=state.get("action_plan_text", "无") or "无",
        )
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        report = resp.choices[0].message.content.strip()
    except Exception as e:
        # 兜底：拼接各模块文本
        parts = []
        if state.get("diagnosis_summary"):
            parts.append(f"**问题诊断**\n{state['diagnosis_summary']}")
        if state.get("recommendation_text"):
            parts.append(f"**商品推荐**\n{state['recommendation_text']}")
        if state.get("action_plan_text"):
            parts.append(f"**行动计划**\n{state['action_plan_text']}")
        report = "\n\n".join(parts) or "分析完成，请查看各模块结果。"

    return {
        "final_report": report,
        "trajectory": [_record("report_generator", time.time() - t0, "报告生成完成")],
    }
