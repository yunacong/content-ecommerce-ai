"""
行动计划生成器。

输入：归因结果 + 推荐 SKU 列表
输出：结构化任务列表（ActionTask），直接可写入任务看板

设计原则：
- 规则先行，LLM 增强（避免硬依赖 LLM 影响稳定性）
- 每个推荐 SKU 至少生成 1 个高优先级任务
- 归因结果决定任务类型偏向
"""
from __future__ import annotations

import json
from typing import Optional
from openai import OpenAI
from loguru import logger

from .task_schema import ActionTask, task_to_dict
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


# 问题类型 → 默认任务模板
_PROBLEM_TASK_TEMPLATES: dict[str, dict] = {
    "CVR下降": {
        "task_type": "内容优化",
        "action": "优化商品详情页信任背书：增加真实评价截图、使用场景图、成分安全说明",
        "owner": "内容运营",
        "deadline": "3天内",
        "expected_metric": "CVR",
        "review_metric": "CVR / 加购率",
    },
    "CTR下降": {
        "task_type": "内容优化",
        "action": "测试痛点型标题（A/B 测试至少2个版本），更新封面主图突出核心卖点",
        "owner": "内容运营",
        "deadline": "2天内",
        "expected_metric": "CTR",
        "review_metric": "CTR / 点击量",
    },
    "曝光下降": {
        "task_type": "投放调整",
        "action": "扩大投放人群包，补充关键词覆盖，检查内容是否触发限流",
        "owner": "投放运营",
        "deadline": "今天",
        "expected_metric": "曝光量",
        "review_metric": "曝光 / CPM",
    },
    "客单价下降": {
        "task_type": "商品运营",
        "action": "测试套装组合定价，内容突出「高性价比」或「品质感」，避免低价内卷",
        "owner": "商品运营",
        "deadline": "5天内",
        "expected_metric": "AOV",
        "review_metric": "AOV / 退款率",
    },
    "综合指标下滑": {
        "task_type": "数据监控",
        "action": "全链路排查：曝光→点击→加购→下单各环节漏斗，锁定最大断点后专项优化",
        "owner": "数据分析",
        "deadline": "今天",
        "expected_metric": "GMV",
        "review_metric": "GMV / 转化漏斗",
    },
}

_PRIORITY_MAP = {
    "CVR下降": "高",
    "CTR下降": "高",
    "曝光下降": "高",
    "客单价下降": "中",
    "综合指标下滑": "高",
}


def _rule_based_tasks(
    diagnosis: dict,
    recommended_skus: list[dict],
) -> list[ActionTask]:
    """基于规则生成任务列表（快速、稳定）。"""
    problem_type = diagnosis.get("problem_type", "综合指标下滑")
    affected_category = diagnosis.get("affected_category", "")
    tmpl = _PROBLEM_TASK_TEMPLATES.get(problem_type, _PROBLEM_TASK_TEMPLATES["综合指标下滑"])
    priority = _PRIORITY_MAP.get(problem_type, "中")

    tasks: list[ActionTask] = []

    # 每个推荐 SKU 生成一条主任务
    for i, sku in enumerate(recommended_skus[:5]):  # 最多前5个 SKU
        sku_id = sku.get("sku_id", f"SKU_{i+1}")
        sku_name = sku.get("product_name", sku_id)[:20]
        content_angle = sku.get("content_angle", "")

        action = tmpl["action"]
        if content_angle:
            action += f"；内容方向建议：{content_angle}"

        expected_impact = (
            f"针对 {sku_name}，通过{tmpl['task_type']}提升{tmpl['expected_metric']}"
        )

        tasks.append(ActionTask(
            task_id=f"T{i+1:03d}",
            task_name=f"【{problem_type}】{sku_name} - {tmpl['task_type']}",
            related_sku=sku_id,
            task_type=tmpl["task_type"],
            priority=priority if i == 0 else ("高" if priority == "高" and i < 2 else "中"),
            expected_metric=tmpl["expected_metric"],
            expected_impact=expected_impact,
            action=action,
            owner=tmpl["owner"],
            deadline=tmpl["deadline"],
            review_metric=tmpl["review_metric"],
        ))

    # 额外补一条数据监控任务
    tasks.append(ActionTask(
        task_id=f"T{len(tasks)+1:03d}",
        task_name=f"持续监控{affected_category or '全品类'} {tmpl['expected_metric']} 变化",
        related_sku="ALL",
        task_type="数据监控",
        priority="中",
        expected_metric=tmpl["expected_metric"],
        expected_impact="及时发现优化效果，决定是否继续放量或调整策略",
        action=f"每日查看 {tmpl['review_metric']}，连续3天改善则扩大投放，下降则暂停调整",
        owner="数据分析",
        deadline="持续",
        review_metric=tmpl["review_metric"],
    ))

    return tasks


def _llm_enhance_tasks(
    tasks: list[ActionTask],
    diagnosis: dict,
    client: OpenAI,
) -> list[ActionTask]:
    """用 LLM 对规则任务进行语言润色和补充（可选增强）。"""
    prompt = f"""你是内容电商运营专家。以下是根据经营诊断自动生成的行动任务，请对每条任务的「action」字段进行优化，使其更具体、可执行，同时不改变任务结构。

经营诊断：
{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

任务列表：
{json.dumps([task_to_dict(t) for t in tasks], ensure_ascii=False, indent=2)}

请返回优化后的 action 字段列表（JSON 数组，顺序与输入一致），每条不超过60字。
只返回 JSON，不要额外解释。"""

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        content = resp.choices[0].message.content.strip()
        # 提取 JSON 数组
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            actions = json.loads(content[start:end])
            for task, new_action in zip(tasks, actions):
                if isinstance(new_action, str) and new_action.strip():
                    task.action = new_action.strip()
    except Exception as e:
        logger.warning(f"LLM 任务增强失败（使用规则版本）: {e}")

    return tasks


def generate_action_plan(
    diagnosis: dict,
    recommended_skus: list[dict],
    use_llm: bool = True,
    llm_client: Optional[OpenAI] = None,
) -> list[dict]:
    """
    生成完整行动计划。

    Args:
        diagnosis: 来自 metric_attribution.attribution_to_dict() 的归因结果
                   必须包含 problem_type, affected_category
        recommended_skus: 来自推荐服务的 SKU 列表
                          每个 SKU 至少包含 sku_id, product_name, content_angle
        use_llm: 是否用 LLM 增强任务描述
        llm_client: 可复用外部已初始化的 OpenAI client

    Returns:
        任务字典列表，可直接渲染到 Streamlit 或写入数据库
    """
    tasks = _rule_based_tasks(diagnosis, recommended_skus)

    if use_llm and DEEPSEEK_API_KEY:
        client = llm_client or OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        tasks = _llm_enhance_tasks(tasks, diagnosis, client)

    return [task_to_dict(t) for t in tasks]


def format_action_plan_text(tasks: list[dict]) -> str:
    """将任务列表格式化为可读文本，供 Agent 最终回复使用。"""
    if not tasks:
        return "暂无行动计划。"

    lines = [f"**行动计划（共 {len(tasks)} 项任务）**\n"]
    for t in tasks:
        lines.append(
            f"[{t['priority']}优先] {t['task_name']}\n"
            f"  执行动作：{t['action']}\n"
            f"  负责人：{t['owner']} | 截止：{t['deadline']} | 复盘指标：{t['review_metric']}\n"
        )
    return "\n".join(lines)
