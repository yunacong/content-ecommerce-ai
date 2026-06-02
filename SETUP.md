# 快速启动指南

## 1. 安装依赖（约 3-5 分钟）

```bash
cd ~/Desktop/content_ecommerce_ai
pip install -r requirements.txt
```

## 2. 填入 DeepSeek API Key

编辑 `.env` 文件：
```
DEEPSEEK_API_KEY=你的API Key
```

## 3. 准备数据（< 1 分钟，纯模拟数据）

```bash
python scripts/prepare_data.py --mode demo
```

## 4. 构建向量索引（约 3-5 分钟，BGE-M3 推理）

**本地 CPU（推荐 demo 模式）：**
```bash
python scripts/build_index.py --mode demo
```

**AutoDL GPU（完整版，含 CLIP 图片向量）：**
```bash
python scripts/build_index.py --mode full
```

## 5. 启动 Demo

```bash
streamlit run app/main.py
```

浏览器打开 http://localhost:8501

---

## 6. 运行消融实验（可选，面试展示用）

```bash
python scripts/run_evaluation.py
```

结果保存在 `data/processed/ablation_results.csv`，在评估看板页面可视化展示。

---

## 项目结构

```
content_ecommerce_ai/
├── project2_infra/          # Project 2：检索推荐底座
│   ├── embedder.py          # BGE-M3 + CLIP 向量化
│   ├── hybrid_search.py     # BM25 + BGE + RRF 混合检索
│   ├── reranker.py          # BGE Reranker cross-encoder
│   ├── multimodal_retrieval.py  # CLIP+BERT 多模态案例检索
│   ├── lightgbm_ranker.py   # LightGBM 排序 + SHAP 解释
│   └── evaluator.py         # Recall@K / MRR / NDCG 评估体系
│
├── project1_app/            # Project 1：AI 经营增长平台
│   ├── text_to_sql.py       # Text-to-SQL + 根因分析
│   ├── content_diagnosis.py # CLIP 封面评分 + BGE 标题评分
│   ├── rag_knowledge.py     # RAG 知识库 + 引用溯源
│   ├── product_selection.py # AI 选品（调用 Project 2）
│   └── agent.py             # ReAct Agent（手写循环）
│
├── scripts/
│   ├── prepare_data.py      # 生成模拟数据 + SQLite DB
│   ├── build_index.py       # 构建 FAISS 索引
│   └── run_evaluation.py    # 消融实验
│
└── app/                     # Streamlit Demo
    ├── main.py
    └── pages/
        ├── agent_chat.py        # AI 经营助手对话
        ├── data_analysis.py     # 数据分析
        ├── content_diagnosis.py # 内容诊断
        ├── product_selection.py # 选品推荐
        ├── knowledge_base.py    # 知识库问答
        └── evaluation_dashboard.py  # 消融实验看板
```
