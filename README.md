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

本项目包含两个互补模块，共同构成内容电商商家 AI 增长闭环：

| | 项目一：AI 经营增长平台 | 项目二：检索推荐底座 |
|---|---|---|
| **定位** | 前台业务应用 | 后台底层能力 |
| **职责** | 问问题 + 生成建议 | 找证据 + 排优先级 |
| **核心技术** | RAG + Function Calling Agent | 混合检索 + Reranker + LightGBM |

## 系统架构

```
商家提问：「为什么本周 GMV 下滑？有没有可参考爆款案例？」
                            │
         ┌──────────────────▼──────────────────┐
         │     项目一：AI 经营增长平台（前台）     │
         │  Agent → Text-to-SQL → 数据分析      │
         │         → RAG → 平台规则/SOP          │
         │         → 内容诊断 → CLIP+BGE 评分    │
         └──────────────────┬──────────────────┘
                            │ 调用
         ┌──────────────────▼──────────────────┐
         │     项目二：检索推荐底座（后台）        │
         │  混合召回：BGE-M3 + BM25 + RRF       │
         │  精排：BGE Reranker cross-encoder    │
         │  排序：LightGBM + 多模态特征           │
         │  解释：LLM + SHAP 证据引用             │
         └──────────────────────────────────────┘
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

### LightGBM 特征消融

| 特征组 | AUC | NDCG@5 |
|--------|-----|--------|
| A. 纯结构化（冷启动） | 0.702 | 0.921 |
| B. +文本语义+CTR | **0.999** | **1.000** |
| C. +多模态图片 | 0.999 | 1.000 |

A→B 跳升：**AUC +42%**，文本语义是关键判别信号

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
content_ecommerce_ai/
├── project2_infra/        # Project 2：检索推荐底座
│   ├── embedder.py        # BGE-M3 + CLIP 向量化
│   ├── hybrid_search.py   # BM25 + BGE + RRF 混合检索
│   ├── reranker.py        # BGE Reranker cross-encoder
│   ├── multimodal_retrieval.py  # CLIP+BERT 多模态检索
│   ├── lightgbm_ranker.py # LightGBM 排序 + SHAP
│   └── evaluator.py       # Recall@K / MRR / NDCG
├── project1_app/          # Project 1：AI 经营增长平台
│   ├── text_to_sql.py     # Text-to-SQL + 根因分析
│   ├── content_diagnosis.py  # CLIP 封面 + BGE 标题评分
│   ├── rag_knowledge.py   # RAG 知识库 + 引用溯源
│   ├── product_selection.py  # AI 选品（调用 Project 2）
│   └── agent.py           # Function Calling Agent
├── scripts/               # 数据准备 + 评估脚本
├── app/                   # Streamlit Demo
└── INTERVIEW_GUIDE.md     # 面试话术指南
```

## Demo 功能

- 🤖 **AI 经营助手**：Function Calling Agent，多工具编排，真实数据分析
- 📊 **经营数据分析**：Text-to-SQL，自然语言→SQL→图表→AI洞察
- 🎨 **内容诊断**：BGE 标题语义评分 + LLM 脚本分析
- 🛍️ **AI 选品推荐**：混合检索→Reranker→LightGBM→SHAP可解释
- 📚 **运营知识库**：RAG + BGE Reranker + 引用来源溯源
- 🔬 **检索评估看板**：消融实验可视化，指标对比

---

*目标公司：腾讯云 / 阿里云 / 火山云及同类云厂商*
*适用岗位：AI 解决方案 / To B 产品经理 / 售前顾问*
