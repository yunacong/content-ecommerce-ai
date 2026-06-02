"""
数据准备脚本：
1. 生成模拟电商经营数据库（SQLite，用于 Text-to-SQL）
2. 构建演示用商品数据（Amazon Reviews 2023 Beauty 子集 or 模拟数据）
3. 构建爆款案例库
4. 构建 RAG 知识库文档
5. 构建评估集

运行：python scripts/prepare_data.py [--mode demo|full]
  demo: 纯模拟数据，<1min，推荐本地快速启动
  full: 下载 Amazon Reviews 2023，需要较长时间（建议 AutoDL）
"""
import sys
import argparse
import sqlite3
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RAW_DIR, PROCESSED_DIR, KNOWLEDGE_DIR, EVAL_DIR


CATEGORIES = ["美妆护肤", "个护清洁", "彩妆", "香水", "男士护肤"]
PLATFORMS = ["抖音", "小红书", "淘宝"]

SKU_NAMES = [
    "玻尿酸精华液", "氨基酸洗面奶", "防晒霜SPF50+", "保湿面霜", "眼霜淡纹",
    "美白面膜套装", "卸妆油温和型", "口红哑光", "气垫BB霜", "睫毛膏防水",
    "香水女士花果香", "男士护肤套装", "祛痘精华", "收缩毛孔爽肤水", "头发护理精油",
    "粉底液持久遮瑕", "腮红橘色", "眉笔防水", "唇釉镜面", "定妆喷雾",
]


def create_directories():
    for d in [RAW_DIR, PROCESSED_DIR, KNOWLEDGE_DIR, EVAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def build_sqlite_db():
    """生成 90 天经营数据，用于 Text-to-SQL 演示"""
    db_path = PROCESSED_DIR / "ecommerce.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS daily_metrics")
    c.execute("""CREATE TABLE daily_metrics (
        date TEXT, sku_id TEXT, sku_name TEXT, category TEXT,
        gmv REAL, impressions INT, clicks INT, add_to_cart INT, orders INT,
        ctr REAL, cvr REAL, aov REAL
    )""")

    c.execute("DROP TABLE IF EXISTS content_metrics")
    c.execute("""CREATE TABLE content_metrics (
        date TEXT, content_id TEXT, sku_id TEXT, platform TEXT, title TEXT,
        views INT, likes INT, comments INT, shares INT, ctr REAL
    )""")

    rows_metrics = []
    rows_content = []
    base_date = datetime.now() - timedelta(days=90)

    for day_offset in range(90):
        dt = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        # 模拟周末流量高，近期GMV下滑趋势
        weekend_boost = 1.3 if (base_date + timedelta(days=day_offset)).weekday() >= 5 else 1.0
        trend = 1.0 - 0.002 * max(0, day_offset - 60)  # 最近30天下滑

        for i, sku_name in enumerate(SKU_NAMES):
            sku_id = f"SKU{i+1:03d}"
            cat = CATEGORIES[i % len(CATEGORIES)]
            base_imp = random.randint(3000, 15000)
            impressions = int(base_imp * weekend_boost * trend * random.uniform(0.8, 1.2))
            ctr = random.uniform(0.03, 0.12)
            clicks = int(impressions * ctr)
            cvr = random.uniform(0.02, 0.08)
            orders = int(clicks * cvr)
            add_to_cart = int(clicks * random.uniform(0.15, 0.35))
            aov = random.uniform(50, 300)
            gmv = orders * aov
            rows_metrics.append((dt, sku_id, sku_name, cat, round(gmv, 2), impressions,
                                  clicks, add_to_cart, orders, round(ctr, 4), round(cvr, 4), round(aov, 2)))

            if random.random() < 0.3:  # 30% 的SKU有当日内容
                platform = random.choice(PLATFORMS)
                views = int(impressions * random.uniform(0.1, 0.5))
                rows_content.append((
                    dt, f"CNT{day_offset*100+i}", sku_id, platform,
                    f"{sku_name}真实测评｜{random.choice(['亲测有效', '踩雷警告', '平价替代'])}",
                    views, int(views * 0.05), int(views * 0.01), int(views * 0.02),
                    round(random.uniform(0.02, 0.15), 4)
                ))

    c.executemany("INSERT INTO daily_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows_metrics)
    c.executemany("INSERT INTO content_metrics VALUES (?,?,?,?,?,?,?,?,?,?)", rows_content)
    conn.commit()
    conn.close()
    logger.info(f"SQLite DB built: {db_path} ({len(rows_metrics)} rows)")


CATEGORY_SKUS = {
    "美妆护肤": ["玻尿酸精华液", "保湿面霜", "眼霜淡纹", "美白面膜套装", "祛痘精华", "收缩毛孔爽肤水", "防晒霜SPF50+"],
    "个护清洁": ["氨基酸洗面奶", "卸妆油温和型", "头发护理精油", "男士护肤套装"],
    "彩妆": ["口红哑光", "气垫BB霜", "睫毛膏防水", "粉底液持久遮瑕", "腮红橘色", "眉笔防水", "唇釉镜面", "定妆喷雾"],
    "香水": ["香水女士花果香"],
    "男士护肤": ["男士护肤套装", "氨基酸洗面奶", "保湿面霜"],
}

def build_product_catalog(n: int = 500) -> list:
    """构建演示用商品目录（类目与商品名匹配）"""
    products = []
    adjectives = ["温和", "高效", "经典", "轻薄", "持久", "保湿", "清爽", "滋润", "修护", "提亮"]
    for i in range(n):
        cat = random.choice(CATEGORIES)
        base_name = random.choice(CATEGORY_SKUS.get(cat, SKU_NAMES))
        title = f"{random.choice(adjectives)}{base_name} {random.choice(['升级版', '特护型', '精华型', ''])}".strip()
        products.append({
            "id": f"P{i:04d}",
            "title": title,
            "text": f"{title}。类目：{cat}。适用肤质：{random.choice(['干性', '油性', '混合性', '敏感肌', '所有肤质'])}。核心成分：{random.choice(['玻尿酸', '烟酰胺', '视黄醇', '神经酰胺', '维C'])}。",
            "category": cat,
            "price": round(random.uniform(29, 398), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "review_count": random.randint(50, 5000),
            "ctr": round(random.uniform(0.03, 0.15), 4),
            "cvr": round(random.uniform(0.02, 0.08), 4),
        })
    path = PROCESSED_DIR / "products.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    logger.info(f"Product catalog: {path} ({n} products)")
    return products


def build_case_library(n: int = 200) -> list:
    """构建爆款案例库"""
    case_templates = [
        "【{adj}测评】{sku}真的绝了！{benefit}",
        "{sku}使用30天后，皮肤{result}",
        "平价版{sku}｜{price}元{contrast}",
        "敏感肌慎选！{sku}{warning}",
        "学生党必备｜{sku}性价比分析",
    ]
    adj_list = ["素人", "真实", "深度", "颠覆认知"]
    benefits = ["上脸超滋润", "毛孔细了好多", "暗沉拜拜", "防晒还透气"]
    results = ["白了两个色号", "出油减少50%", "毛孔几乎看不见", "细纹淡化了"]
    contrasts = ["秒杀大牌", "平替神器", "学生也买得起"]

    cases = []
    for i in range(n):
        sku = random.choice(SKU_NAMES)
        cat = CATEGORIES[SKU_NAMES.index(sku) % len(CATEGORIES)] if sku in SKU_NAMES else CATEGORIES[0]
        template = random.choice(case_templates)
        title = template.format(
            adj=random.choice(adj_list), sku=sku, benefit=random.choice(benefits),
            result=random.choice(results), price=random.randint(29, 199),
            contrast=random.choice(contrasts), warning="不踩坑指南"
        )
        # 爆款特征：高CTR
        ctr = random.uniform(0.08, 0.25)
        cases.append({
            "id": f"CASE{i:04d}",
            "title": title,
            "text": title + f" 类目:{cat}",
            "category": cat,
            "platform": random.choice(PLATFORMS),
            "ctr": round(ctr, 4),
            "cvr": round(random.uniform(0.05, 0.15), 4),
            "views": random.randint(10000, 500000),
            "cover_path": None,  # 实际项目中存图片路径
        })
    path = PROCESSED_DIR / "cases.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logger.info(f"Case library: {path} ({n} cases)")
    return cases


def build_knowledge_docs() -> list:
    """构建 RAG 知识库文档"""
    docs = [
        {
            "id": "rule_douyin",
            "title": "抖音电商运营规则2024",
            "source": "抖音电商运营规则2024",
            "doc_type": "platform_rule",
            "content": """抖音电商平台运营规范（2024版）

一、商品标题规范
商品标题不得超过60个字符，禁止使用夸大性词语如"最便宜"、"第一"等绝对化表述。
标题应包含品牌名、商品名称、核心特性，避免堆砌关键词。

二、直播间规范
直播过程中禁止虚假宣传，所有功效宣称需有相应资质证明。
促销活动需在直播开始前24小时报备，价格不得低于平台最低价保障线。
禁止引流至站外平台，包括微信、微博等第三方社交平台。

三、内容创作规范
短视频内容不得含有虚假评论或刷量行为。
医疗类产品宣传需持有相关资质，美妆类产品不得宣称医疗功效。
违规内容将被下架，情节严重者封禁账号。

四、售后服务标准
商家需在买家发起退款后48小时内处理，超时将自动退款。
七天无理由退换货适用于大多数类目，特殊类目需标注说明。""",
        },
        {
            "id": "sop_content",
            "title": "爆款内容创作SOP",
            "source": "爆款内容创作SOP",
            "doc_type": "sop",
            "content": """爆款内容创作标准操作流程

一、选题阶段
1. 分析近30天同类目爆款内容，提取共同选题规律
2. 结合当前热点话题和季节性需求
3. 优先选择高搜索量、低竞争度的长尾关键词

二、封面设计原则
封面是决定点击率的最关键因素，建议：
- 主体突出：人脸或商品特写占画面60%以上
- 文字精简：不超过10个字，字号大于画面宽度的1/8
- 色彩对比：背景与主体色彩对比度高，避免杂乱
- 对比图：before/after效果图点击率比普通图高40%

三、标题公式
高CTR标题通常包含：痛点词+解决方案+数字+情绪词
示例：敏感肌必看｜3步去红去痘印，皮肤科医生推荐配方

四、脚本结构
- 0-3秒：钩子（制造悬念/共鸣/冲突）
- 3-30秒：核心价值展示
- 30-55秒：佐证（数据/对比/用户评价）
- 55-60秒：行动号召（下方链接/评论关键词）

五、发布时间
美妆类：晚上8-10点发布效果最佳
护肤类：周末上午10-12点互动率最高""",
        },
        {
            "id": "guide_selection",
            "title": "内容电商选品方法论",
            "source": "内容电商选品方法论",
            "doc_type": "guide",
            "content": """内容电商选品核心方法论

一、选品核心指标
1. 内容适配性：商品是否有视觉卖点，适合视频展示
2. 价格带：100-300元为内容电商最佳价格带，决策成本低
3. 复购潜力：消耗品优于一次性商品
4. 竞争烈度：避开红海品类，寻找蓝海机会

二、冷启动评估（新品无历史数据）
1. 竞品分析：同类目Top10商品近30天销量和内容数量
2. 搜索趋势：用平台搜索热词工具验证需求
3. 小样测试：先投100-500元测试CTR，CTR>5%再加大投入

三、季节性选品日历
- Q1（1-3月）：年货、情人节、春季护肤
- Q2（4-6月）：防晒、清洁、618大促
- Q3（7-9月）：防晒续购、秋季保湿
- Q4（10-12月）：双11、双12、年末礼盒

四、放弃选品信号
- 连续7天CTR<3%
- ROI<1:1.5
- 用户评价中多次出现同一投诉点""",
        },
        {
            "id": "faq_operation",
            "title": "运营常见问题解答",
            "source": "运营常见问题解答",
            "doc_type": "faq",
            "content": """运营常见问题 FAQ

Q: GMV突然下滑怎么排查？
A: 按以下顺序排查：
1. 检查流量是否下降（曝光数变化）
2. 检查CTR是否下降（流量正常但点击少 → 封面/标题问题）
3. 检查CVR是否下降（点击正常但不下单 → 价格/详情页/评价问题）
4. 检查是否有平台算法调整（查看官方公告）
5. 检查竞争对手是否有大促活动

Q: 如何提升封面点击率？
A:
- A/B测试：同一商品做3个不同封面，跑3天选CTR最高的
- 参考爆款：找同类目CTR>10%的视频，分析其封面构图
- 加文字钩子：封面加"避雷"、"必买"等情绪词可提升CTR 15-20%

Q: 新品如何快速起量？
A:
1. 免费流量：发布后1小时内完播率>50%可获得更多推荐
2. 达人合作：找腰部达人（10-100万粉）性价比最高
3. 千川投放：先跑自然流量再投付费，避免内容质量分低

Q: 平台违规如何申诉？
A: 登录商家后台 → 违规记录 → 申诉中心 → 提交申诉材料（资质证明+整改说明），一般3个工作日内处理。""",
        },
    ]

    for doc in docs:
        path = KNOWLEDGE_DIR / f"{doc['id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    logger.info(f"Knowledge docs built: {len(docs)} documents in {KNOWLEDGE_DIR}")
    return docs


def build_eval_sets(products: list, cases: list):
    """构建评估集：RAG问答集 + 搜索相关性集"""
    # RAG 评估集
    rag_eval = [
        {"query": "抖音直播间可以引导用户加微信吗？", "relevant_docs": ["rule_douyin"], "answer_hint": "不可以"},
        {"query": "爆款标题应该怎么写？", "relevant_docs": ["sop_content"], "answer_hint": "公式"},
        {"query": "新品没有历史数据怎么选品？", "relevant_docs": ["guide_selection"], "answer_hint": "冷启动"},
        {"query": "CTR下降了怎么排查原因？", "relevant_docs": ["faq_operation"], "answer_hint": "排查步骤"},
        {"query": "封面设计有什么原则？", "relevant_docs": ["sop_content"], "answer_hint": "主体突出"},
        {"query": "商品标题最多多少字？", "relevant_docs": ["rule_douyin"], "answer_hint": "60个字符"},
        {"query": "什么价格段的商品最适合做内容电商？", "relevant_docs": ["guide_selection"], "answer_hint": "100-300元"},
        {"query": "七天无理由退换货怎么处理？", "relevant_docs": ["rule_douyin"], "answer_hint": "48小时"},
    ]

    # 搜索评估集（用商品标题构造查询）
    search_eval = []
    for i, product in enumerate(random.sample(products, min(30, len(products)))):
        # 把标题改写成自然语言查询
        query = product["title"].replace("版", "").replace("型", "")
        relevant_ids = [product["id"]]
        # 找同类目的商品也作为相关
        same_cat = [p["id"] for p in products if p["category"] == product["category"] and p["id"] != product["id"]]
        relevant_ids.extend(same_cat[:3])

        search_eval.append({
            "query": query,
            "relevant_ids": [products.index(p) for p in products if p["id"] in relevant_ids],
            "relevance_map": {products.index(p): (2 if p["id"] == product["id"] else 1)
                              for p in products if p["id"] in relevant_ids},
        })

    with open(EVAL_DIR / "rag_eval.json", "w", encoding="utf-8") as f:
        json.dump(rag_eval, f, ensure_ascii=False, indent=2)
    with open(EVAL_DIR / "search_eval.json", "w", encoding="utf-8") as f:
        json.dump(search_eval, f, ensure_ascii=False, indent=2)

    logger.info(f"Eval sets built: {len(rag_eval)} RAG + {len(search_eval)} search queries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    args = parser.parse_args()

    create_directories()
    logger.info(f"Preparing data (mode={args.mode})")

    build_sqlite_db()
    products = build_product_catalog(n=500 if args.mode == "demo" else 2000)
    cases = build_case_library(n=200 if args.mode == "demo" else 500)
    build_knowledge_docs()
    build_eval_sets(products, cases)

    logger.info("✅ Data preparation complete!")
    logger.info(f"Next step: python scripts/build_index.py --mode {args.mode}")


if __name__ == "__main__":
    main()
