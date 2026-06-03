"""
指标归因树：拆解 GMV 变化的主因，定位问题 SKU / 品类 / 指标。

GMV = 曝光 × CTR × CVR × 客单价

输出结构化归因结果，供 Agent 生成推荐请求和行动计划。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from scipy import stats


@dataclass
class AttributionResult:
    main_problem: str                    # e.g. "CVR下降"
    problem_metric: str                  # "cvr" / "ctr" / "impressions" / "aov"
    affected_category: str               # 影响最大的品类
    affected_skus: list[str]             # 影响最大的 SKU 列表
    gmv_change_pct: float                # GMV 变化百分比（负数表示下降）
    metric_changes: dict[str, float]     # 各指标环比变化
    evidence: list[str]                  # 证据句列表，供 LLM / 报告直接引用
    recommendation_request: dict         # 直接可传给推荐服务的请求参数


def _load_period(conn: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM daily_metrics WHERE date BETWEEN ? AND ?",
        conn,
        params=(start, end),
    )


def _agg(df: pd.DataFrame) -> dict:
    """聚合一个周期内的核心指标（加权平均/总量）。"""
    if df.empty:
        return {}
    total_impressions = df["impressions"].sum()
    total_clicks = df["clicks"].sum()
    total_orders = df["orders"].sum()
    total_gmv = df["gmv"].sum()
    return {
        "gmv": total_gmv,
        "impressions": total_impressions,
        "ctr": total_clicks / total_impressions if total_impressions else 0,
        "cvr": total_orders / total_clicks if total_clicks else 0,
        "aov": total_gmv / total_orders if total_orders else 0,
    }


def _pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old


def _find_dragging_skus(
    df_cur: pd.DataFrame, df_prev: pd.DataFrame, top_n: int = 3
) -> tuple[list[str], str]:
    """找到 GMV 贡献下跌最多的 SKU 和品类。"""
    cur_sku = df_cur.groupby(["sku_id", "sku_name", "category"])["gmv"].sum().reset_index()
    prev_sku = df_prev.groupby(["sku_id", "sku_name", "category"])["gmv"].sum().reset_index()

    merged = cur_sku.merge(prev_sku, on=["sku_id", "sku_name", "category"], suffixes=("_cur", "_prev"))
    merged["gmv_drop"] = merged["gmv_prev"] - merged["gmv_cur"]
    merged = merged.sort_values("gmv_drop", ascending=False)

    top_skus = merged.head(top_n)["sku_name"].tolist()
    affected_category = (
        merged.groupby("category")["gmv_drop"].sum().idxmax()
        if not merged.empty else ""
    )
    return top_skus, affected_category


def _identify_main_problem(metric_changes: dict[str, float]) -> tuple[str, str]:
    """根据各指标变化量，判断主要问题来源。"""
    # GMV 分解：ΔGMV ≈ 曝光贡献 + CTR贡献 + CVR贡献 + AOV贡献
    # 简化：取绝对值最大的负向指标
    candidates = {
        "impressions": ("曝光下降", metric_changes.get("impressions", 0)),
        "ctr": ("CTR下降", metric_changes.get("ctr", 0)),
        "cvr": ("CVR下降", metric_changes.get("cvr", 0)),
        "aov": ("客单价下降", metric_changes.get("aov", 0)),
    }
    worst_key = min(candidates, key=lambda k: candidates[k][1])
    worst_label, worst_val = candidates[worst_key]

    # 如果最差指标变化不显著（> -3%），综合判断
    if worst_val > -0.03:
        return "综合指标下滑", worst_key

    return worst_label, worst_key


def _build_evidence(
    metric_changes: dict[str, float],
    gmv_change_pct: float,
    affected_skus: list[str],
    affected_category: str,
) -> list[str]:
    lines = []
    pct = lambda v: f"{v*100:+.1f}%"

    lines.append(f"近期 GMV 环比变化 {pct(gmv_change_pct)}")

    metric_labels = {
        "impressions": "曝光量",
        "ctr": "点击率（CTR）",
        "cvr": "转化率（CVR）",
        "aov": "客单价（AOV）",
    }
    for k, label in metric_labels.items():
        v = metric_changes.get(k, 0)
        if abs(v) > 0.01:
            lines.append(f"{label} 环比 {pct(v)}")

    if affected_category:
        lines.append(f"影响最大的品类：{affected_category}")
    if affected_skus:
        lines.append(f"拖累最大的 SKU：{'、'.join(affected_skus[:3])}")

    return lines


def analyze_gmv_drop(
    db_path: str,
    current_start: str,
    current_end: str,
    prev_start: str,
    prev_end: str,
    target_category: str = "",
    price_range: Optional[list[float]] = None,
) -> AttributionResult:
    """
    对比两个时间段，输出结构化归因结果。

    参数：
        db_path: SQLite 数据库路径
        current_start/end: 当前周期（如近7天）
        prev_start/end: 对比周期（如上一个7天）
        target_category: 可选，限定品类
        price_range: 可选，限定价格区间

    返回：
        AttributionResult，包含 recommendation_request 可直接传给推荐服务
    """
    conn = sqlite3.connect(db_path)
    df_cur = _load_period(conn, current_start, current_end)
    df_prev = _load_period(conn, prev_start, prev_end)
    conn.close()

    if target_category:
        df_cur = df_cur[df_cur["category"] == target_category]
        df_prev = df_prev[df_prev["category"] == target_category]

    agg_cur = _agg(df_cur)
    agg_prev = _agg(df_prev)

    if not agg_cur or not agg_prev:
        return AttributionResult(
            main_problem="数据不足",
            problem_metric="unknown",
            affected_category=target_category,
            affected_skus=[],
            gmv_change_pct=0.0,
            metric_changes={},
            evidence=["当前时间段无数据，无法归因"],
            recommendation_request={},
        )

    metric_changes = {
        k: _pct_change(agg_cur[k], agg_prev[k])
        for k in ("gmv", "impressions", "ctr", "cvr", "aov")
    }
    gmv_change_pct = _pct_change(agg_cur["gmv"], agg_prev["gmv"])

    main_problem, problem_metric = _identify_main_problem(metric_changes)
    affected_skus, affected_category = _find_dragging_skus(df_cur, df_prev)

    if target_category:
        affected_category = target_category

    evidence = _build_evidence(metric_changes, gmv_change_pct, affected_skus, affected_category)

    # 根据归因结果，构造推荐请求参数
    goal_map = {
        "CVR下降": "提升CVR",
        "CTR下降": "提升CTR",
        "曝光下降": "提升曝光",
        "客单价下降": "提升客单价",
        "综合指标下滑": "提升GMV",
    }
    recommendation_request = {
        "business_goal": goal_map.get(main_problem, "提升GMV"),
        "problem_type": main_problem,
        "target_category": affected_category,
        "price_range": price_range or [0, 9999],
        "top_k": 10,
    }

    return AttributionResult(
        main_problem=main_problem,
        problem_metric=problem_metric,
        affected_category=affected_category,
        affected_skus=affected_skus,
        gmv_change_pct=gmv_change_pct,
        metric_changes=metric_changes,
        evidence=evidence,
        recommendation_request=recommendation_request,
    )


def attribution_to_dict(result: AttributionResult) -> dict:
    return {
        "main_problem": result.main_problem,
        "problem_metric": result.problem_metric,
        "affected_category": result.affected_category,
        "affected_skus": result.affected_skus,
        "gmv_change_pct": result.gmv_change_pct,
        "metric_changes": result.metric_changes,
        "evidence": result.evidence,
        "recommendation_request": result.recommendation_request,
    }
