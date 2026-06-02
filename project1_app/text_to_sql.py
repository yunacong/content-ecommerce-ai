"""Text-to-SQL：自然语言 → SQL → 执行 → 图表 + 根因分析"""
import sqlite3
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DATA_DIR


SCHEMA_DESC = """
数据库表结构：

表 daily_metrics（每日经营指标）:
- date TEXT           -- 日期 YYYY-MM-DD
- sku_id TEXT         -- 商品ID
- sku_name TEXT       -- 商品名称
- category TEXT       -- 类目
- gmv REAL            -- GMV（元）
- impressions INT     -- 曝光数
- clicks INT          -- 点击数
- add_to_cart INT     -- 加购数
- orders INT          -- 订单数
- ctr REAL            -- 点击率 = clicks/impressions
- cvr REAL            -- 转化率 = orders/clicks
- aov REAL            -- 客单价 = gmv/orders

表 content_metrics（内容表现指标）:
- date TEXT
- content_id TEXT     -- 内容ID
- sku_id TEXT         -- 关联商品ID
- platform TEXT       -- 平台 (抖音/小红书/淘宝)
- title TEXT          -- 内容标题
- views INT           -- 播放量
- likes INT           -- 点赞数
- comments INT        -- 评论数
- shares INT          -- 分享数
- ctr REAL            -- 内容点击率
"""

FEW_SHOT = [
    {
        "question": "本周GMV和上周相比下降了多少？",
        "sql": "SELECT strftime('%W', date) as week, SUM(gmv) as total_gmv FROM daily_metrics GROUP BY week ORDER BY week DESC LIMIT 2"
    },
    {
        "question": "哪个SKU点击率最低？",
        "sql": "SELECT sku_name, AVG(ctr) as avg_ctr FROM daily_metrics GROUP BY sku_id, sku_name ORDER BY avg_ctr ASC LIMIT 5"
    },
    {
        "question": "最近7天转化漏斗各环节数据",
        "sql": "SELECT SUM(impressions) as 曝光, SUM(clicks) as 点击, SUM(add_to_cart) as 加购, SUM(orders) as 下单 FROM daily_metrics WHERE date >= date('now', '-7 days')"
    },
]


class TextToSQL:
    def __init__(self, db_path: Optional[str] = None):
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._db_path = db_path or str(DATA_DIR / "processed" / "ecommerce.db")

    def _get_connection(self):
        return sqlite3.connect(self._db_path)

    def generate_sql(self, question: str) -> str:
        few_shot_text = "\n".join(
            f"问题：{ex['question']}\nSQL：{ex['sql']}" for ex in FEW_SHOT
        )
        prompt = f"""你是一个电商数据分析专家，将自然语言问题转换为 SQLite SQL 查询。

{SCHEMA_DESC}

示例：
{few_shot_text}

规则：
1. 只输出 SQL，不要解释
2. 使用 SQLite 语法
3. 日期用 date('now', '-N days') 表示相对日期
4. 结果加中文列别名方便展示
5. 【重要】只生成一条 SQL 语句，不要用分号分隔多条语句
6. 如果问题涉及多个维度，用 JOIN 或子查询合并在一条 SQL 里

问题：{question}
SQL："""

        resp = self._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        sql = resp.choices[0].message.content.strip()
        sql = re.sub(r"```sql\n?|```", "", sql).strip()
        # 防御：只取第一条语句（遇到分号截断）
        if ";" in sql:
            sql = sql.split(";")[0].strip()
        return sql

    def execute_sql(self, sql: str) -> Tuple[pd.DataFrame, Optional[str]]:
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return df, None
        except Exception as e:
            return pd.DataFrame(), str(e)

    def generate_insight(self, question: str, df: pd.DataFrame) -> str:
        if df.empty:
            return "查询结果为空，无法生成洞察。"

        data_summary = df.to_string(index=False, max_rows=20)
        prompt = f"""你是一个内容电商经营顾问。根据以下数据，用3-5句话给出核心洞察和可执行建议。

商家问题：{question}

数据结果：
{data_summary}

要求：
- 直接指出最重要的问题或机会
- 给出具体可执行的建议（不要说"加强"、"优化"这种空话）
- 如有异常数据，给出可能的根因假设
- 语言简洁，面向非技术运营人员"""

        resp = self._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    def auto_chart(self, df: pd.DataFrame, question: str) -> Optional[go.Figure]:
        """根据数据结构自动选择图表类型"""
        if df.empty or len(df.columns) < 2:
            return None

        cols = df.columns.tolist()
        date_cols = [c for c in cols if "date" in c.lower() or "week" in c.lower() or "日期" in c]
        num_cols = [c for c in cols if df[c].dtype in ["float64", "int64"] and c not in date_cols]

        is_funnel = any(kw in question for kw in ["漏斗", "funnel", "曝光", "转化链路"])
        is_single_row = len(df) == 1

        if is_funnel and is_single_row and len(num_cols) >= 3:
            # 单行汇总数据 → 漏斗图
            funnel_cols = [c for c in num_cols if df[c].iloc[0] > 0]
            funnel_vals = [int(df[c].iloc[0]) for c in funnel_cols]
            fig = go.Figure(go.Funnel(
                y=funnel_cols, x=funnel_vals,
                textposition="inside", textinfo="value+percent initial",
            ))
            fig.update_layout(title=question)
        elif is_single_row and len(num_cols) >= 2:
            # 单行多列 → 水平柱状图
            plot_df = df[num_cols].T.reset_index()
            plot_df.columns = ["指标", "数值"]
            fig = px.bar(plot_df, x="数值", y="指标", orientation="h", title=question)
        elif date_cols and num_cols:
            fig = px.line(df, x=date_cols[0], y=num_cols[0], title=question,
                         labels={date_cols[0]: "日期", num_cols[0]: num_cols[0]})
        elif len(num_cols) == 1 and len(cols) == 2:
            cat_col = [c for c in cols if c not in num_cols][0]
            fig = px.bar(df.head(10), x=cat_col, y=num_cols[0], title=question)
        else:
            fig = px.bar(df.head(15), x=cols[0], y=num_cols[0] if num_cols else cols[1], title=question)

        fig.update_layout(height=400, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    def query(self, question: str) -> Dict:
        """端到端：问题 → SQL → 执行 → 图表 → 洞察"""
        sql = self.generate_sql(question)
        df, error = self.execute_sql(sql)

        if error:
            return {"success": False, "sql": sql, "error": error}

        insight = self.generate_insight(question, df)
        chart = self.auto_chart(df, question)

        return {
            "success": True,
            "sql": sql,
            "data": df,
            "insight": insight,
            "chart": chart,
        }
