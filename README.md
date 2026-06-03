---
title: 内容电商商家 AI 运营助手
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# 内容电商商家 AI 运营助手 | RAG · LangGraph Agent · Text-to-SQL · CLIP

> 面向内容电商商家的 ToB AI 运营助手，覆盖经营诊断、商品推荐、内容优化、行动计划的完整增长闭环

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 项目概述

面向内容电商商家的 ToB AI 运营助手，覆盖从经营诊断到行动计划的完整闭环。

与 [多模态商品选品推荐系统](https://github.com/yunacong/multimodal-recsys) 配合，形成：

| 本项目（AI 运营助手） | 配套项目（推荐服务） |
|---|---|
| 经营诊断、Agent 编排、行动计划 | 多模态召回排序、B 端推荐 API |
| LangGraph / Function Calling | Two-Tower + LightGBM + FastAPI |

## 系统架构

```
商家提问：「最近 GMV 为什么下降？帮我推荐重点商品并给出行动计划」
                            │
         ┌──────────────────▼──────────────────────┐
         │   LangGraph / Function Calling Agent    │
         └──────────────────┬──────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    Text-to-SQL        RAG 知识库        内容诊断
    经营数据查询       平台规则/SOP      CLIP+BGE评分
           │
           ▼
    GMV 指标归因树
    曝光×CTR×CVR×AOV
           │
           ▼
    多模态选品推荐服务            ← 可调用 multimodal-recsys API
    /merchant/recommend
           │
           ▼
    商品评论洞察
    卖点/痛点/内容角度
           │
           ▼
    行动计划生成
    任务/负责人/截止/复盘指标
           │
           ▼
    最终经营建议报告
```

## 技术栈

| 层级 | 技术 | 用途 |
|---|---|---|
| 大模型 | DeepSeek (Function Calling) | Agent、Text-to-SQL、RAG 生成 |
| 向量检索 | BGE-M3 + FAISS | 稠密向量检索 |
| 稀疏检索 | BM25 (rank-bm25) | 关键词检索 |
| 融合 | RRF | 混合召回融合 |
| 精排 | BGE Reranker v2-m3 | cross-encoder 重排 |
| 多模态 | CLIP ViT-B/32 | 封面图向量、以图搜货 |
| 排序模型 | LightGBM + SHAP | Learning-to-Rank |
| 工程 | Python + Streamlit + SQLite | Demo 前端 |

## 消融实验结果

### 检索消融（Project 2）

| 配置 | Recall@5 | MRR | NDCG@10 |
|------|----------|-----|---------|
| ① BM25 基线 | 0.183 | 0.451 | 0.297 |
| ② BGE 稠密向量 | 0.300 | 0.774 | 0.509 |
| ③ Hybrid BM25+BGE+RRF | 0.275 | 0.695 | 0.440 |
| **④ Hybrid + BGE Reranker** | **0.300** | **0.793** | **0.513** |

最优配置 vs 基线：**MRR +75.8%，NDCG@10 +72.9%**

### LightGBM 排序实验说明

本项目使用 LightGBM + SHAP 作为商品候选排序和可解释推荐理由生成的核心模块。

早期在模拟数据上进行的特征消融实验中，加入历史 CTR/CVR 特征后 AUC 接近 1，经复盘判断主要来自**模拟标签规则过强**和**历史点击率特征存在目标泄露风险**，该结果不作为真实业务效果指标。

当前项目中 LightGBM 的核心价值在于：

| 功能 | 说明 |
|------|------|
| 候选 SKU 排序 | 按经营目标（提升GMV/CVR/CTR）对候选商品重排序 |
| SHAP 特征贡献 | 解释每个推荐结果的主要驱动因素 |
| 推荐理由生成 | 将特征贡献转化为可读的选品理由 |
| 经营目标映射 | 根据问题类型调整特征权重 |

> 真实泛化指标参见 [multimodal-recsys](https://github.com/yunacong/multimodal-recsys) 中的严格评估结果（修复时序泄露后 AUC 0.609）。

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 3. 准备数据（< 1 分钟）

```bash
python scripts/prepare_data.py --mode demo
```

### 4. 构建向量索引（约 5 分钟，首次下载模型）

```bash
python scripts/build_index.py --mode demo
```

### 5. 启动 Demo

```bash
./start.sh
# 浏览器打开 http://localhost:8501
```

### 6. 公网访问（需要 ngrok）

```bash
# 安装: brew install ngrok
# 配置: ngrok config add-authtoken <token>
./deploy_public.sh
```

### 7. 运行消融实验（可选，约 5 分钟）

```bash
python scripts/train_ranker.py       # LightGBM 训练
python scripts/run_evaluation.py     # 检索消融实验
```

## 项目结构

```
content-ecommerce-ai/
├── project1_app/
│   ├── agent.py                        # Function Calling Agent（DeepSeek）
│   ├── agent_graph/                    # LangGraph 多节点状态机
│   │   ├── graph.py                    # StateGraph 编排 + 条件路由
│   │   ├── nodes.py                    # 8 个独立节点函数
│   │   └── state.py                    # AgentState 共享状态
│   ├── diagnosis/
│   │   └── metric_attribution.py       # GMV = 曝光×CTR×CVR×AOV 归因树
│   ├── tools/
│   │   ├── merchant_rec_tool.py        # 双模式 B 端选品推荐（本地/远程）
│   │   └── review_insight_tool.py      # 商品评论洞察工具
│   ├── action_plan/
│   │   ├── task_schema.py              # ActionTask 数据结构
│   │   └── task_generator.py           # 行动计划生成（规则+LLM增强）
│   ├── text_to_sql.py                  # Text-to-SQL + 图表生成
│   ├── rag_knowledge.py                # RAG 知识库 + 引用溯源
│   └── content_diagnosis.py            # CLIP+BGE 内容诊断评分
├── project2_infra/                     # 本地检索推荐底座
│   ├── embedder.py                     # BGE-M3 + CLIP 向量化
│   ├── hybrid_search.py                # BM25 + BGE + RRF 混合检索
│   ├── reranker.py                     # BGE Reranker cross-encoder
│   ├── lightgbm_ranker.py              # LightGBM 排序 + SHAP
│   └── evaluator.py                    # Recall@K / MRR / NDCG
├── app/                                # Streamlit Demo（7 页面）
├── scripts/                            # 数据准备 + 评估脚本
└── INTEGRATION_DEMO.md                 # 端到端集成演示
```

## Demo 页面

- 🔗 **LangGraph AI 助手**：多节点状态图，流式展示执行进度
- 🤖 **Function Calling 助手**：原始 Agent，多工具调用
- 📊 **经营数据分析**：Text-to-SQL → 图表 → AI 洞察
- 🛍️ **AI 选品推荐**：混合检索 → Reranker → LightGBM → SHAP 可解释
- 📚 **运营知识库**：RAG + BGE Reranker + 引用来源溯源
- 🎨 **内容诊断**：BGE 标题评分 + LLM 脚本分析
- 🔬 **检索评估看板**：消融实验可视化

## 适用岗位

- AI 应用开发实习生
- 大模型应用开发实习生（RAG / Agent / LangGraph）
- 电商 AI / 商业化 AI 工程实习生
- 数据智能应用实习生
- AI 解决方案 / ToB AI 产品实习生
