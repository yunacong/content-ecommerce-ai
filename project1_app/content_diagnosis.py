"""智能内容诊断：封面图视觉评分（CLIP）+ 标题语义评分（BGE）+ LLM 改进建议"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
import faiss


class ContentDiagnosis:
    def __init__(self, embedder=None, case_retriever=None):
        self._embedder = embedder        # Embedder instance
        self._case_retriever = case_retriever  # MultimodalCaseRetriever
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def _has_image_index(self) -> bool:
        """检查是否有图片向量索引（full 模式才有）"""
        if self._case_retriever is None:
            return False
        from pathlib import Path
        from config import PROCESSED_DIR
        return Path(str(PROCESSED_DIR / "cases") + "_image.index").exists()

    def score_cover(self, cover_image: Image.Image, top_k: int = 5) -> Dict:
        """封面图评分：与爆款案例库对比，输出视觉相似度分"""
        if self._embedder is None or self._case_retriever is None:
            return {"score": 0.5, "similar_cases": [], "mode": "unavailable",
                    "tip": "模型未加载"}

        # demo 模式下没有图片向量索引，改用 LLM 视觉分析
        if not self._has_image_index():
            return self._score_cover_llm(cover_image)

        try:
            img_vec = self._embedder.encode_image_from_pil([cover_image])
            similar = self._case_retriever.search_by_image(img_vec[0], top_k=top_k)
        except Exception as e:
            return self._score_cover_llm(cover_image, fallback_reason=str(e))

        if not similar:
            return {"score": 0.3, "similar_cases": [], "mode": "clip"}

        avg_ctr = np.mean([c.get("ctr", 0.05) for c in similar])
        top_score = similar[0].get("image_score", 0.5)
        score = min(top_score * (avg_ctr / 0.10), 1.0)

        return {
            "score": float(score),
            "top_similarity": float(top_score),
            "benchmark_ctr": float(avg_ctr),
            "similar_cases": similar[:3],
            "mode": "clip",
        }

    def _score_cover_llm(self, cover_image: Image.Image, fallback_reason: str = "") -> Dict:
        """
        当图片向量索引不可用时（demo 模式），改用 LLM 多模态视觉分析封面。
        将图片转为 base64 发给 DeepSeek-VL / 通用 LLM 分析视觉质量。
        """
        import base64, io
        buf = io.BytesIO()
        cover_image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = """你是内容电商封面图审核专家。请分析这张封面图，从以下维度评分（0-10分）并给出改进建议。

评分维度：
1. 主体突出度：主体（人物/商品）是否清晰占主画面
2. 视觉冲击力：色彩对比、构图是否吸引眼球
3. 文字清晰度：文字是否简洁易读（无文字也可以）
4. 情绪感染力：是否能引发用户点击欲

输出 JSON 格式（只输出 JSON，不要其他）：
{"主体突出度": 7, "视觉冲击力": 6, "文字清晰度": 8, "情绪感染力": 7,
 "overall": 7.0, "issues": ["问题1"], "suggestions": ["建议1", "建议2"]}"""

        try:
            resp = self._client.chat.completions.create(
                model=self._get_vision_model(),
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=0.2,
                max_tokens=300,
            )
            import json, re
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json\n?|```", "", raw).strip()
            result = json.loads(raw)
            overall = result.get("overall", 6.0)
            score = float(overall) / 10.0
            return {
                "score": score,
                "mode": "llm_vision",
                "dimensions": {k: v for k, v in result.items() if k not in ("overall", "issues", "suggestions")},
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "tip": "当前使用 LLM 视觉分析（demo 模式）；full 模式可启用 CLIP 向量对标",
            }
        except Exception as e:
            return {
                "score": 0.5,
                "mode": "llm_vision_failed",
                "tip": f"LLM 视觉分析失败：{str(e)}。请确认 API Key 支持多模态。",
            }

    def _get_vision_model(self) -> str:
        """DeepSeek 多模态模型名"""
        import os
        return os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-chat")

    def score_title(self, title: str, category: str = "", top_k: int = 5) -> Dict:
        """标题评分：与爆款案例标题做语义距离计算"""
        if self._embedder is None or self._case_retriever is None:
            return {"score": 0.5, "similar_titles": []}

        query_vec = self._embedder.encode_text([title])
        similar = self._case_retriever.search_by_text(query_vec[0], top_k=top_k)

        if not similar:
            return {"score": 0.3, "similar_titles": []}

        top_score = similar[0].get("text_score", 0.5)
        avg_ctr = np.mean([c.get("ctr", 0.05) for c in similar])
        score = min(top_score * (avg_ctr / 0.10), 1.0)

        return {
            "score": float(score),
            "top_similarity": float(top_score),
            "benchmark_ctr": float(avg_ctr),
            "similar_titles": [{"title": c.get("title", ""), "ctr": c.get("ctr", 0)} for c in similar[:3]],
        }

    def analyze_script(self, script: str) -> Dict:
        """脚本结构分析：钩子、卖点、行动号召"""
        prompt = f"""分析以下短视频脚本的结构质量，输出JSON格式评估结果。

脚本内容：
{script[:2000]}

请分析以下维度（每项0-10分）：
1. hook_score: 开头3秒钩子强度（能否留住用户）
2. selling_point_score: 卖点表达完整性（核心卖点是否清晰）
3. cta_score: 行动号召有效性（是否有明确引导购买/互动）
4. structure_score: 整体结构流畅度

同时给出：
- issues: 主要问题列表（最多3个）
- suggestions: 具体改进建议（最多3个，要具体可操作）

只输出JSON，不要其他文字。格式：
{{"hook_score": 7, "selling_point_score": 6, "cta_score": 5, "structure_score": 7,
  "issues": ["问题1", "问题2"], "suggestions": ["建议1", "建议2"]}}"""

        resp = self._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        import json, re
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json\n?|```", "", raw).strip()
        try:
            result = json.loads(raw)
        except Exception:
            result = {"hook_score": 5, "selling_point_score": 5, "cta_score": 5, "structure_score": 5,
                     "issues": ["解析失败"], "suggestions": [raw[:200]]}
        return result

    def full_diagnosis(
        self,
        title: str,
        cover_image: Optional[Image.Image] = None,
        script: Optional[str] = None,
        category: str = "",
    ) -> Dict:
        """综合诊断报告"""
        report = {"title": title, "category": category}

        title_result = self.score_title(title, category)
        report["title_diagnosis"] = title_result
        report["title_score"] = title_result["score"]

        if cover_image:
            cover_result = self.score_cover(cover_image)
            report["cover_diagnosis"] = cover_result
            report["cover_score"] = cover_result["score"]
        else:
            report["cover_score"] = None

        if script:
            script_result = self.analyze_script(script)
            report["script_diagnosis"] = script_result
            avg = np.mean([
                script_result.get("hook_score", 5),
                script_result.get("selling_point_score", 5),
                script_result.get("cta_score", 5),
            ]) / 10.0
            report["script_score"] = float(avg)
        else:
            report["script_score"] = None

        # 综合得分
        scores = [s for s in [report["title_score"], report.get("cover_score"), report.get("script_score")] if s is not None]
        report["overall_score"] = float(np.mean(scores)) if scores else 0.5

        return report
