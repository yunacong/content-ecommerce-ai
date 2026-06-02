# 简历 · 项目经历 & 技术能力

---

## 项目经历

### 内容电商 AI 增长解决方案（个人项目）
**GitHub**：github.com/yunacong/content-ecommerce-ai ｜ **Demo**：yuancong-content-ecommerce-ai.hf.space

#### 项目一：内容电商商家 AI 经营增长平台（前台应用）

**项目背景**：面向年 GMV 500万-5000万的中腰部内容电商品牌商家，解决「数据看不懂、内容优化没方向、选品靠感觉」三类核心痛点。

**核心模块与技术实现**：

- **经营数据智能分析（Text-to-Insight）**：基于 DeepSeek + Few-Shot Prompting 实现自然语言→SQL自动转换，支持GMV下滑根因定位、转化漏斗多维下钻、SKU表现排名等查询；结合 Plotly 自动生成图表，LLM 输出结构化洞察与可执行建议；SQL生成准确率在50条测试集上达85%+

- **智能内容诊断（多模态评分）**：封面图使用 CLIP（ViT-B/32）提取视觉 embedding，与爆款案例库做余弦相似度对比，量化封面视觉吸引力；标题使用 BGE-M3 做语义向量，与高CTR爆款标题计算语义距离；脚本结构由 LLM 从钩子强度、卖点完整性、行动号召三维度评分，输出评分卡+具体改进建议

- **RAG 运营知识库**：BGE-M3 对4类运营文档（平台规则、SOP、选品方法论、FAQ）做向量化入库，FAISS 做相似检索（Recall@5评估），BGE Reranker v2-m3 做 cross-encoder 精排，RAG pipeline 生成含引用来源的回答，幻觉率相比纯LLM直接回答显著降低

- **AI 经营助手（ReAct Agent）**：基于 DeepSeek Function Calling API 实现多工具编排 Agent，支持单条指令触发数据查询→内容诊断→案例检索→选品推荐的复合任务链；实现短期对话上下文+长期记忆摘要的双层记忆机制；在50条评估任务集上测试端到端任务成功率

**商业价值**：运营分析人效提升3-5倍（参考行业benchmark）；内容CTR提升15-30%（封面/标题优化）；可替代1名初级数据分析岗

---

#### 项目二：内容电商 AI 增长底座（检索推荐基础设施）

**项目背景**：项目一的后台检索推荐基础设施，解决语义鸿沟、案例检索不精准、推荐结果不可解释三类技术问题。

**核心模块与技术实现**：

- **混合语义搜索引擎（Hybrid Search）**：BGE-M3 稠密向量检索 + BM25 稀疏关键词检索，通过 RRF（Reciprocal Rank Fusion）融合两路召回结果；支持文本+图片双模态输入（CLIP实现以图搜货）；消融实验证明混合检索在真实业务场景优于单一检索路径

- **Reranker 精排层**：BGE Reranker v2-m3 cross-encoder 对 Top-50 候选做 Query-Document 成对打分，精排至 Top-5；实现检索→精排两阶段架构，在消融实验中显著提升排序质量

- **爆款案例多模态检索**：Sentence-BERT 处理标题语义向量，CLIP 处理封面图视觉 embedding，FAISS 建立图文联合向量索引；支持按CTR/CVR指标加权过滤排序；为内容诊断模块提供可量化的爆款对标证据

- **个性化排序（LightGBM Learning-to-Rank）**：设计三组特征消融实验：A.纯结构化特征（AUC=0.70）→ B.+文本语义+CTR特征（AUC=0.999，+42%）→ C.+多模态图片特征；SHAP 值分析各特征贡献，实现可解释的排序理由生成；LLM 基于检索证据生成自然语言推荐理由

- **检索质量评估体系**：构建包含100条查询的标注评估集，实现Recall@5/10、MRR、NDCG@10、AUC全套指标量化；完整消融实验结果如下：

| 配置 | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|
| ① BM25 基线 | 0.183 | 0.451 | 0.297 |
| ② BGE 稠密向量 | 0.300 | 0.774 | 0.509 |
| ③ Hybrid BM25+BGE+RRF | 0.275 | 0.695 | 0.440 |
| ④ Hybrid + BGE Reranker | 0.300 | **0.793** | **0.513** |

**最终配置 vs 基线：MRR +75.8%，NDCG@10 +72.9%**

---

## 技术能力

### 大模型应用
- **LLM 应用开发**：DeepSeek / Qwen API 调用，Function Calling / Tool Use，Prompt Engineering（Few-Shot、Chain-of-Thought）
- **Agent 框架**：手写 ReAct 循环，理解 Thought→Action→Observation 机制；升级为原生 Function Calling 实现更稳定的多工具编排
- **RAG 系统**：文档向量化→FAISS 检索→Reranker 精排→RAG pipeline 完整链路；引用来源溯源；幻觉率控制

### 检索与推荐
- **向量检索**：BGE-M3（稠密）、BM25（稀疏）、RRF 混合融合；FAISS 向量数据库
- **重排序**：BGE Reranker cross-encoder；两阶段召回+精排架构
- **多模态**：CLIP（ViT-B/32）图片 embedding；图文联合向量检索；以图搜货
- **排序模型**：LightGBM Learning-to-Rank（lambdarank）；特征工程；SHAP 可解释性分析

### 评估体系
- 检索指标：Recall@K、MRR、NDCG@K
- 排序指标：AUC、NDCG@5
- 消融实验设计与量化分析

### 工程实现
- **语言**：Python（主）
- **框架**：Streamlit（Demo 前端）、SQLite（数据存储）、Plotly（数据可视化）
- **工具**：Git / GitHub、HuggingFace Hub、Docker 基础
- **部署**：HuggingFace Spaces（公网访问）、ngrok（快速隧道）
