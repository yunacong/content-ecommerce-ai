"""Agent：使用 DeepSeek Function Calling API，工具调用更稳定可靠"""
import json
from typing import List, Dict, Optional
from openai import OpenAI
from loguru import logger
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


# OpenAI-compatible function definitions（DeepSeek 原生 Function Calling 格式）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "查询电商经营数据，分析GMV、CTR、CVR、转化漏斗、SKU排名等指标。当用户问到数据、下滑原因、表现分析时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "具体的数据分析问题，例如：本周GMV与上周对比"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_content",
            "description": "诊断内容质量，分析标题语义相似度、脚本结构等，给出量化评分和改进建议。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "视频标题"},
                    "script": {"type": "string", "description": "脚本内容（可选）"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cases",
            "description": "检索爆款案例库，找到高CTR的历史成功内容案例。当用户要参考案例、找爆款时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索描述，如：美妆护肤爆款案例"},
                    "category": {"type": "string", "description": "商品类目（可选）"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_products",
            "description": "AI选品推荐，为内容投放筛选高潜力商品并排序，给出推荐理由。当用户要选品、找商品时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "选品需求描述"},
                    "category": {"type": "string", "description": "类目限定（可选）"},
                    "budget": {"type": "number", "description": "预算上限（可选）"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": "查询运营知识库，回答平台规则、SOP、运营技巧等问题。当用户问规则、方法论时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "具体问题"},
                },
                "required": ["question"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个专业的内容电商 AI 经营助手，帮助商家解决运营增长问题。

你有以下工具可以调用：
- query_data：查询经营数据（GMV、CTR、CVR、漏斗等）
- diagnose_content：诊断内容质量（标题、封面、脚本）
- search_cases：检索爆款案例
- select_products：AI选品推荐
- query_knowledge：查询运营知识库（平台规则、SOP）

工作原则：
1. 对于需要数据支撑的问题，先调用工具获取真实数据，再给出分析
2. 回答要包含具体数字，不说空话
3. 给出1-3条可立即执行的建议
4. 多步骤问题可以依次调用多个工具"""


class EcommerceAgent:
    def __init__(self, text_to_sql=None, content_diagnosis=None, rag_kb=None,
                 product_selector=None, case_retriever=None, embedder=None):
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._text_to_sql = text_to_sql
        self._content_diagnosis = content_diagnosis
        self._rag_kb = rag_kb
        self._product_selector = product_selector
        self._case_retriever = case_retriever
        self._embedder = embedder

        self._history: List[Dict] = []   # 长期记忆
        self._short_ctx: List[Dict] = [] # 短期上下文

    def _call_tool(self, tool_name: str, params: Dict) -> str:
        try:
            if tool_name == "query_data" and self._text_to_sql:
                # Agent 场景只做 SQL 执行，不单独调 LLM 生成 insight（Agent 最终合成时会综合分析）
                sql = self._text_to_sql.generate_sql(params["question"])
                df, error = self._text_to_sql.execute_sql(sql)
                if error:
                    return f"查询失败: {error}"
                data_str = df.to_string(index=False, max_rows=15) if not df.empty else "无数据"
                return f"SQL: {sql}\n\n查询结果:\n{data_str}"

            elif tool_name == "diagnose_content" and self._content_diagnosis:
                result = self._content_diagnosis.full_diagnosis(
                    title=params.get("title", ""),
                    script=params.get("script"),
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "search_cases" and self._case_retriever and self._embedder:
                query = params["query"]
                vec = self._embedder.encode_text([query])[0]
                cases = self._case_retriever.search_by_text(vec, top_k=5)
                if not cases:
                    return "未找到相关爆款案例"
                lines = [f"- {c.get('title', 'N/A')} | CTR:{c.get('ctr', 0):.2%} | 类目:{c.get('category', 'N/A')}"
                         for c in cases]
                return f"找到 {len(cases)} 个爆款案例：\n" + "\n".join(lines)

            elif tool_name == "select_products" and self._product_selector:
                result = self._product_selector.select(
                    query=params["query"],
                    category=params.get("category", ""),
                    budget=params.get("budget"),
                    top_k=5,
                )
                products = result["candidates"]
                if not products:
                    return "未找到匹配商品"
                lines = []
                for i, p in enumerate(products):
                    line = f"{i+1}. {p.get('title', 'N/A')} | 评分:{p.get('rating', 0):.1f} | CTR:{p.get('ctr_hist', p.get('ctr', 0)):.2%}"
                    if p.get('recommendation_reason'):
                        line += f"\n   推荐理由: {p['recommendation_reason']}"
                    lines.append(line)
                return f"推荐 {len(products)} 个高潜力商品：\n" + "\n".join(lines)

            elif tool_name == "query_knowledge" and self._rag_kb:
                result = self._rag_kb.answer(params["question"])
                return f"知识库回答:\n{result['answer']}\n\n来源: {', '.join(result['sources'])}"

            else:
                return f"工具 [{tool_name}] 暂不可用"

        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            return f"工具执行出错: {str(e)}"

    def _build_memory_context(self) -> str:
        if not self._history:
            return ""
        items = self._history[-5:]
        lines = "\n".join(f"- {m['summary']}" for m in items)
        return f"\n\n【历史对话摘要】\n{lines}"

    def _extract_memory(self, user_msg: str, trajectory: list) -> None:
        for step in trajectory:
            result_text = str(step.get("result", ""))
            tool = step.get("tool", "")
            if len(result_text) > 20 and tool in ("query_data", "select_products"):
                summary = f"[{tool}] 「{user_msg[:20]}」→ {result_text[:80]}..."
                self._history.append({"tool": tool, "query": user_msg, "summary": summary})

    def chat(self, user_message: str, max_steps: int = 6) -> Dict:
        """使用 Function Calling API 进行多轮工具调用"""
        system = SYSTEM_PROMPT + self._build_memory_context()
        messages = [{"role": "system", "content": system}]
        messages.extend(self._short_ctx[-6:])
        messages.append({"role": "user", "content": user_message})

        trajectory = []

        for step in range(max_steps):
            resp = self._client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1000,
            )
            msg = resp.choices[0].message

            # 有工具调用
            if msg.tool_calls:
                # 手动构造 dict，避免 Pydantic model_dump by_alias 版本兼容问题
                tool_calls_dict = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tool_calls_dict,
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        params = json.loads(tc.function.arguments)
                    except Exception:
                        params = {}

                    logger.info(f"Step {step+1}: Calling {tool_name}({params})")
                    observation = self._call_tool(tool_name, params)
                    trajectory.append({
                        "step": step + 1,
                        "tool": tool_name,
                        "params": params,
                        "result": observation,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    })
                continue

            # 没有工具调用 = 最终答案
            final_answer = (msg.content or "").strip()
            break
        else:
            final_answer = "已完成数据查询，请查看上方工具调用结果。"

        self._extract_memory(user_message, trajectory)
        self._short_ctx.append({"role": "user", "content": user_message})
        self._short_ctx.append({"role": "assistant", "content": final_answer})

        return {
            "answer": final_answer,
            "trajectory": trajectory,
            "steps": len(trajectory),
        }

    def reset_context(self, clear_memory: bool = False):
        self._short_ctx = []
        if clear_memory:
            self._history = []

    @property
    def memory_summary(self) -> List[str]:
        return [m["summary"] for m in self._history[-5:]]
