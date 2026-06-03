"""行动计划任务的数据结构定义。"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ActionTask:
    task_id: str
    task_name: str
    related_sku: str
    task_type: Literal["内容优化", "投放调整", "商品运营", "数据监控"]
    priority: Literal["高", "中", "低"]
    expected_metric: str          # 预期影响的指标，如 "CTR" / "CVR" / "GMV"
    expected_impact: str          # 预期影响描述
    action: str                   # 具体执行动作
    owner: str                    # 建议负责人
    deadline: str                 # 建议完成时间
    review_metric: str            # 复盘时关注的指标
    status: Literal["待执行", "执行中", "已完成", "已放弃"] = "待执行"


def task_to_dict(t: ActionTask) -> dict:
    return {
        "task_id": t.task_id,
        "task_name": t.task_name,
        "related_sku": t.related_sku,
        "task_type": t.task_type,
        "priority": t.priority,
        "expected_metric": t.expected_metric,
        "expected_impact": t.expected_impact,
        "action": t.action,
        "owner": t.owner,
        "deadline": t.deadline,
        "review_metric": t.review_metric,
        "status": t.status,
    }
