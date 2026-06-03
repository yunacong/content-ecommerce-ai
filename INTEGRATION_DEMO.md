# 端到端集成演示：GMV 下滑 → 选品推荐 → 行动计划

本文档说明如何从一个真实商家问题出发，走完"经营诊断 → 商品推荐 → 评论洞察 → 行动计划"的完整闭环。

---

## 场景设定

> 商家输入：「最近 7 天 GMV 为什么下降？帮我推荐接下来重点推的商品，并给出行动计划。」

---

## 完整链路（LangGraph 版本）

```
商家提问
   ↓
[节点1] intent_router
   识别意图 = "diagnosis+recommendation"
   提取参数：problem_type=综合指标下滑, business_goal=提升GMV
   ↓
[节点2] data_diagnosis（Text-to-SQL）
   生成 SQL → 查询近7天 GMV / CTR / CVR / 曝光 变化
   返回：CVR 从 4.8% 下降至 3.5%，面膜品类拖累最大
   ↓
[节点3] metric_attribution（指标归因树）
   GMV = 曝光 × CTR × CVR × 客单价
   定位主因：CVR下降，受影响品类=面膜，受影响 SKU=SKU_023/SKU_041
   输出 recommendation_request 直接传给推荐服务
   ↓
[节点4] product_recommendation（多模态选品推荐）
   调用 merchant_recommend_products(business_goal=提升CVR, problem_type=CVR下降)
   → 本地 project2_infra 排序 或 远程 /merchant/recommend API
   返回 Top-5 SKU + 推荐理由 + 内容方向 + 风险提示
   ↓
[节点5] review_insight（商品评论洞察）
   对 Top-1 SKU 调用 get_review_insight
   提取：补水效果好、敏感肌友好（卖点）/ 精华液偏少（痛点）
   内容角度：熬夜急救补水 / 敏感肌温和修护
   ↓
[节点6] action_planner（行动计划生成）
   规则先行 + LLM 增强，生成可执行任务：
   T001: [高] 优化 SKU_023 详情页信任背书，内容角度=熬夜急救补水，3天内，复盘 CVR
   T002: [中] 暂停 SKU_041 低效投放素材，今天，复盘 ROI
   T003: [中] 持续监控面膜品类 CVR 变化，持续
   ↓
[节点7] report_generator（最终报告）
   汇总以上所有节点输出，生成结构化经营建议报告
```

---

## 运行方式

### 方式一：LangGraph Agent（推荐）

```python
from project1_app.agent_graph.graph import build_graph, run_graph
from project1_app.text_to_sql import TextToSQL
from project1_app.rag_knowledge import RAGKnowledgeBase

ts = TextToSQL()
rag = RAGKnowledgeBase()
graph = build_graph(text_to_sql=ts, rag_kb=rag, db_path="data/processed/ecommerce.db")

result = run_graph(graph, "最近7天GMV为什么下降？帮我推荐接下来重点推的商品")
print(result["final_report"])
print(result["action_tasks"])
```

### 方式二：原始 Function Calling Agent

```python
from project1_app.agent import EcommerceAgent
from project1_app.text_to_sql import TextToSQL

agent = EcommerceAgent(text_to_sql=TextToSQL())
result = agent.chat("最近7天GMV为什么下降？帮我推荐接下来重点推的商品")
print(result["answer"])
```

### 方式三：Streamlit Demo

```bash
streamlit run streamlit_app.py
# 选择左侧「🔗 AI 助手 · LangGraph」页面
```

---

## 与多模态推荐服务的集成

项目二（[multimodal-recsys](https://github.com/yunacong/multimodal-recsys)）提供独立的 B 端推荐 API，可作为远程服务调用：

```bash
# 启动项目二推荐服务
cd ../multimodal-recsys/serving
uvicorn app.main:app --port 8000

# 项目一切换为远程模式
export USE_REMOTE_RECOMMENDER=true
export RECOMMENDER_API_URL=http://localhost:8000
```

切换后，`merchant_recommend_products` 工具自动走远程 `/merchant/recommend` 接口，其余逻辑不变。

---

## 各模块独立调用

```python
# 1. 指标归因
from project1_app.diagnosis.metric_attribution import analyze_gmv_drop
result = analyze_gmv_drop("data/processed/ecommerce.db",
                          "2026-05-27", "2026-06-03",
                          "2026-05-20", "2026-05-26")
print(result.evidence)
print(result.recommendation_request)

# 2. 评论洞察
from project1_app.tools.review_insight_tool import get_review_insight
print(get_review_insight({"sku_id": "SKU_023"}))

# 3. 行动计划
from project1_app.action_plan.task_generator import generate_action_plan, format_action_plan_text
tasks = generate_action_plan(
    diagnosis={"problem_type": "CVR下降", "affected_category": "面膜"},
    recommended_skus=[{"sku_id": "SKU_023", "product_name": "补水面膜", "content_angle": "熬夜急救"}]
)
print(format_action_plan_text(tasks))
```
