"""
Language-specific configuration for the distill pipeline.
Each language has its own data source, prompts, templates, and keyword rules.
"""

I18N = {
    "zh": {
        "source_file": "analysis.json",
        "output_suffix": "zh",
        "skill_name": "matchmaking-insight-zh",

        # Step 1: normalize labels
        "tier_map": {
            "tier_1": "一线", "tier_1.5": "新一线", "tier_1_5": "新一线",
            "new_first_tier": "新一线", "tier_2": "二线", "tier_3": "三线",
            "tier_4": "四线", "tier_5": "五线", "overseas": "海外",
            "any": "不限", "unknown": "未知", None: "未知",
        },
        "income_labels": {
            "low": "低收入(A4及以下)", "mid": "中等(A5)",
            "mid_high": "中高(A6)", "high": "高收入(A7)", "top": "极高(A8+)",
            "unknown": "未知",
        },
        "edu_keywords": {
            "phd": ["博士", "PhD", "博后"],
            "master": ["硕士", "研究生", "MBA", "硕"],
            "bachelor": ["本科", "大学", "985", "211", "一本", "二本"],
            "below": ["大专", "专科", "中专", "高中", "初中", "肄业", "自考"],
        },
        "edu_labels": {"phd": "博士", "master": "硕士", "bachelor": "本科", "below": "大专及以下", "unknown": "未知"},

        # Step 2: conflict pattern keywords (Chinese)
        "conflict_keyword_regex": r'[一-鿿]{2,4}',
        "pattern_rules": {
            "经济价值vs情绪价值错配": ["经济", "情绪价值", "情商", "赚钱", "老实", "无趣"],
            "自身条件vs期望落差": ["眼高手低", "既要又要", "不匹配", "落差", "错配", "高攀"],
            "慕强心理困境": ["慕强", "强势", "优秀", "向上", "高价值"],
            "年龄焦虑": ["大龄", "年龄", "剩女", "剩男", "催婚"],
            "圈层固化": ["体制内", "圈层", "阶层", "门当户对", "向下兼容"],
            "性格缺陷": ["内耗", "低能量", "被动", "逃避", "自卑", "社恐"],
            "自由vs稳定矛盾": ["自由", "稳定", "丁克", "不婚", "散漫"],
        },
        "other_label": "其他",

        # Step 3: LLM prompts
        "prompt_1": """你是一个婚恋市场分析专家。基于以下{total}条真实相亲案例的统计分析报告和案例样本，提炼心智模型和决策启发式。

## 统计报告
{statistics}

## 代表性案例样本（前30条摘要）
{sample_cases}

## 要求

请严格按以下JSON格式输出，不要输出其他内容：

```json
{{
  "mental_models": [
    {{
      "name": "模型名称",
      "one_liner": "一句话描述",
      "evidence": ["证据1（引用具体案例编号和模式）", "证据2"],
      "application": "什么场景下用这个模型",
      "limitation": "什么情况下失效"
    }}
  ],
  "heuristics": [
    {{
      "rule": "规则名称",
      "description": "具体描述",
      "scenario": "适用场景",
      "case_example": "案例佐证"
    }}
  ]
}}
```

要求：
1. 心智模型5-7个，必须从数据中归纳，不是泛泛的道理
2. 决策启发式8-12条，要具体到可执行
3. 所有内容必须有案例数据支撑，不要编造
4. 每条证据和案例佐证要简短（一句话）""",

        "prompt_2": """你是一个婚恋市场分析专家。基于以下{total}条真实相亲案例的统计分析报告和案例样本，提炼人物原型、反模式和表达DNA。

## 统计报告摘要
- 总案例{total}条（男{male}/女{female}），年龄{age_min}-{age_max}，平均{age_avg}岁
- 性格标签Top10: {top_tags}
- 冲突模式: {conflict_patterns}

## 代表性案例样本（前20条摘要）
{sample_cases}

## 要求

请严格按以下JSON格式输出，不要输出其他内容：

```json
{{
  "archetypes": [
    {{
      "name": "原型名称",
      "gender": "male/female",
      "profile": "画像描述（一句话）",
      "core_conflict": "核心矛盾（一句话）",
      "typical_strategy": "策略建议（一句话）"
    }}
  ],
  "anti_patterns": [
    {{
      "name": "反模式名称",
      "description": "描述（一句话）",
      "frequency": "高/中/低",
      "remedy": "破解方法（一句话）"
    }}
  ],
  "expression_dna": {{
    "style": "分析风格描述",
    "sentence_pattern": "句式特征",
    "vocabulary": ["高频词汇1", "高频词汇2", "...最多10个"],
    "tone": "语气特征",
    "taboo": ["禁忌词1", "禁忌词2", "...最多5个"]
  }}
}}
```

要求：
1. 人物原型8-10个，覆盖男女主要类型，每个描述简短
2. 反模式5-8个，来自案例中反复出现的错误
3. 表达DNA基于陈楠的分析风格：直接犀利、数据驱动、不留情面、善用类比
4. 所有描述尽量简短，一句话为主""",

        "digest_prompt": """你是一个数据脱敏专家。请对以下相亲案例进行脱敏摘要处理。

## 脱敏规则
1. 具体城市名 → 用「某一线城市」「某二线城市」等替代
2. 具体职业/公司 → 用泛化描述（体制内/互联网/金融/医疗等）
3. 精确年龄保留（这不构成隐私风险）
4. 收入层级保留（A5/A6等）
5. 核心冲突和策略建议原文保留（这些是分析结论，不含隐私）
6. 性格标签保留
7. 身高体重等具体数值 → 用描述替代（偏矮/中等/偏高）

## 案例数据
{cases_json}

## 输出格式

对每条案例输出如下格式（不要输出其他内容）：

### {first_id}
- **画像**：[性别]，[年龄]岁，[城市层级]，[泛化职业]，[收入层级]，[学历]
- **性格**：[性格标签]
- **冲突**：[核心冲突原文]
- **策略**：[策略建议原文]

（依次输出每条案例，保持case_id顺序）""",

        # Step 4: SKILL.md template pieces
        "tpl": {
            "model_heading": "模型{i}: {name}",
            "one_liner": "一句话", "evidence": "证据", "application": "应用", "limitation": "局限",
            "scenario": "适用场景", "case_example": "案例佐证",
            "profile": "画像", "core_conflict": "核心矛盾", "strategy": "策略建议",
            "frequency_label": "出现频率", "remedy": "破解",
            "title": "婚恋市场洞察 · 认知操作系统",
            "quote": "「婚恋市场没有感情，只有匹配效率。先认清自己的牌，再决定怎么出。」",
            "usage_title": "使用说明",
            "usage_body": "此 Skill 不是情感咨询师，是**婚恋市场分析工具**。它用数据和模型帮你看清：\n- 你在市场里是什么位置\n- 你的核心矛盾是什么\n- 什么策略最可能成功\n\n**不做的事**：不安慰、不鸡汤、不说「你很好只是没遇到对的人」。",
            "workflow_title": "分析工作流（Agentic Protocol）",
            "workflow_intro": "收到用户的个人条件描述后，按以下流程分析：",
            "step1_title": "Step 1: 信息采集",
            "step1_body": "如果用户没有提供完整信息，追问以下关键维度：\n- 性别、年龄、城市\n- 职业、年收入区间\n- 学历、身高体重\n- 性格自评（3个关键词）\n- 择偶核心要求（最多3条）\n- 过往恋爱经历（有/无，几段）",
            "step2_title": "Step 2: 画像定位",
            "step2_body": "将用户信息映射到典型人物原型（见下方），识别最接近的1-2个原型。",
            "step3_title": "Step 3: 市场估值",
            "step3_body": "基于城市层级、收入、学历、外形、年龄五维度，给出市场竞争力评估。\n不打分，而是说清楚「你的牌是什么，哪张牌硬，哪张牌软」。",
            "step4_title": "Step 4: 核心冲突识别",
            "step4_body": "用心智模型分析用户条件与期望之间的结构性矛盾。",
            "step5_title": "Step 5: 策略建议",
            "step5_body": "基于决策启发式，给出具体可执行的匹配策略：\n- 该在哪个池子里找（圈层定位）\n- 该强调什么（优势放大）\n- 该放弃什么（期望校准）\n- 该避免什么（反模式规避）",
            "models_title": "核心心智模型",
            "heuristics_title": "决策启发式",
            "archetypes_title": "典型人物原型",
            "anti_title": "反模式库（常见错误）",
            "dna_title": "表达DNA",
            "dna_intro": "分析时遵循以下风格：",
            "dna_style": "风格", "dna_sentence": "句式", "dna_vocab": "高频词汇", "dna_tone": "语气", "dna_taboo": "禁忌",
            "boundary_title": "诚实边界",
            "boundary_items": [
                "样本偏向抖音平台用户群体（20-38岁、以一二线城市为主），不代表全部婚恋市场",
                "数据采集截止2026年4月，之后的市场变化未覆盖",
                "所有模型和启发式是统计规律，不是铁律——个体差异始终存在",
                "不涉及感情经营、关系维护等婚后议题",
                "不替代专业心理咨询",
            ],
            "sample_bias": "案例中女性样本（{female}条）多于男性（{male}条），男性画像可能不够全面",
            "appendix_title": "附录：数据来源",
            "appendix_platform": "抖音",
            "appendix_blogger": "陈楠（相亲分析系列）",
            "footer": "本 Skill 由数据蒸馏 pipeline 自动生成\n> 数据来源：陈楠相亲分析系列",
            "trigger_desc": "当用户提到「相亲分析」「婚恋市场」「找对象」「匹配策略」「市场价值」时使用。\n  即使用户只是说「帮我分析一下条件」「我该找什么样的」「为什么找不到对象」也应触发。",
            "pattern_lib_title": "冲突模式库",
            "pattern_cases": "代表案例",
            "archive_title": "典型案例索引",
            "archive_headers": "| Case ID | 性别 | 年龄 | 收入 | 核心冲突 |",
            "vocab_sep": "、",
            # references i18n
            "stats_title": "数据统计快照",
            "stats_generated": "基于 {total} 条案例",
            "stats_gender": "性别", "stats_age": "年龄", "stats_city": "城市层级",
            "stats_income": "收入分布", "stats_edu": "学历分布",
            "stats_tags_title": "性格标签 Top {n}",
            "stats_male_tags": "男性标签 Top {n}", "stats_female_tags": "女性标签 Top {n}",
            "stats_cooccur": "标签共现 Top {n}",
            "stats_conflict_kw": "冲突关键词 Top {n}", "stats_strategy_kw": "策略关键词 Top {n}",
            "stats_pattern_dist": "冲突模式分布",
            "pattern_typical": "典型表现",
            "digest_title": "案例脱敏摘要",
            "digest_note": "以下为全部案例的脱敏摘要，已去除可识别个人信息。保留性别、年龄段、收入层级、核心冲突和策略精华。",
        },

        # Step 5: quality check regex
        "qc": {
            "model_re": r'^###\s+模型\d',
            "limit_re": r'局限|失效',
            "archetype_re": r'^###\s+[♂♀]',
            "boundary_section_re": r'##\s+诚实边界(.*?)(?=\n##|\Z)',
            "anti_re": r'^\d+\.\s+\*\*.*?频率',
        },
    },

    "en": {
        "source_file": "analysis_en.json",
        "output_suffix": "en",
        "skill_name": "matchmaking-insight-en",

        "tier_map": {
            "tier_1": "Tier 1", "tier_1.5": "New Tier 1", "tier_1_5": "New Tier 1",
            "new_first_tier": "New Tier 1", "tier_2": "Tier 2", "tier_3": "Tier 3",
            "tier_4": "Tier 4", "tier_5": "Tier 5", "overseas": "Overseas",
            "any": "Any", "unknown": "Unknown", None: "Unknown",
        },
        "income_labels": {
            "low": "Low (A4-)", "mid": "Middle (A5)",
            "mid_high": "Upper-Mid (A6)", "high": "High (A7)", "top": "Top (A8+)",
            "unknown": "Unknown",
        },
        "edu_keywords": {
            "phd": ["PhD", "Doctoral", "Postdoc", "博士", "博后"],
            "master": ["Master", "MBA", "Graduate", "硕士", "研究生", "硕"],
            "bachelor": ["Bachelor", "Undergraduate", "University", "本科", "大学", "985", "211"],
            "below": ["Associate", "Diploma", "High School", "大专", "专科", "中专", "高中", "初中"],
        },
        "edu_labels": {"phd": "PhD", "master": "Master's", "bachelor": "Bachelor's", "below": "Below Bachelor's", "unknown": "Unknown"},

        "conflict_keyword_regex": r'\b[a-zA-Z]{4,}\b',
        "pattern_rules": {
            "Economic vs Emotional Value Mismatch": ["economic", "emotional", "EQ", "earning", "boring", "dull", "value"],
            "Expectation-Reality Gap": ["mismatch", "unrealistic", "gap", "overestimate", "picky"],
            "Hypergamy Trap": ["hypergam", "strong", "superior", "alpha", "high-value"],
            "Age Anxiety": ["age", "older", "leftover", "pressure", "deadline"],
            "Social Class Ceiling": ["class", "tier", "circle", "compatible", "downward"],
            "Personality Deficit": ["passive", "low-energy", "avoidant", "insecure", "anxious"],
            "Freedom vs Stability Conflict": ["freedom", "stable", "childfree", "commitment", "casual"],
        },
        "other_label": "Other",

        "prompt_1": """You are a dating market analyst. Based on the statistical report and case samples from {total} real matchmaking cases below, extract mental models and decision heuristics.

## Statistical Report
{statistics}

## Representative Case Samples (first 30)
{sample_cases}

## Requirements

Output strictly in the following JSON format, nothing else:

```json
{{
  "mental_models": [
    {{
      "name": "Model name",
      "one_liner": "One-line description",
      "evidence": ["Evidence 1 (cite specific case IDs and patterns)", "Evidence 2"],
      "application": "When to apply this model",
      "limitation": "When this model fails"
    }}
  ],
  "heuristics": [
    {{
      "rule": "Rule name",
      "description": "Specific description",
      "scenario": "Applicable scenario",
      "case_example": "Supporting case"
    }}
  ]
}}
```

Requirements:
1. 5-7 mental models, derived from data, not generic wisdom
2. 8-12 heuristics, specific enough to be actionable
3. All content must be supported by case data, no fabrication
4. Keep each evidence and case example brief (one sentence)""",

        "prompt_2": """You are a dating market analyst. Based on the statistical report and case samples from {total} real matchmaking cases, extract archetypes, anti-patterns, and expression DNA.

## Statistical Summary
- Total {total} cases (male {male} / female {female}), age {age_min}-{age_max}, avg {age_avg}
- Top personality tags: {top_tags}
- Conflict patterns: {conflict_patterns}

## Representative Case Samples (first 20)
{sample_cases}

## Requirements

Output strictly in the following JSON format, nothing else:

```json
{{
  "archetypes": [
    {{
      "name": "Archetype name",
      "gender": "male/female",
      "profile": "Profile description (one sentence)",
      "core_conflict": "Core conflict (one sentence)",
      "typical_strategy": "Strategy advice (one sentence)"
    }}
  ],
  "anti_patterns": [
    {{
      "name": "Anti-pattern name",
      "description": "Description (one sentence)",
      "frequency": "High/Medium/Low",
      "remedy": "How to fix (one sentence)"
    }}
  ],
  "expression_dna": {{
    "style": "Analysis style description",
    "sentence_pattern": "Sentence pattern characteristics",
    "vocabulary": ["keyword1", "keyword2", "...up to 10"],
    "tone": "Tone characteristics",
    "taboo": ["taboo1", "taboo2", "...up to 5"]
  }}
}}
```

Requirements:
1. 8-10 archetypes covering major male and female types, keep descriptions brief
2. 5-8 anti-patterns from recurring mistakes in the cases
3. Expression DNA: direct, data-driven, no-nonsense, uses sharp analogies
4. All descriptions should be concise, one sentence each""",

        "digest_prompt": """You are a data de-identification expert. Create de-identified summaries of the following matchmaking cases.

## De-identification Rules
1. Specific city names → use "a tier-1 city", "a tier-2 city", etc.
2. Specific employers/companies → use generalized descriptions (government, tech, finance, medical, etc.)
3. Exact age is kept (not a privacy risk in this context)
4. Income tier is kept (A5/A6 etc.)
5. Core conflict and strategy text is kept as-is (these are analytical conclusions, not PII)
6. Personality tags are kept
7. Specific height/weight → use descriptive terms (short/average/tall)

## Case Data
{cases_json}

## Output Format

For each case, output exactly this format (no other text):

### {first_id}
- **Profile**: [gender], [age], [city tier], [generalized occupation], [income tier], [education]
- **Personality**: [personality tags]
- **Conflict**: [core conflict as-is]
- **Strategy**: [strategy as-is]

(Output each case in order by case_id)""",

        "tpl": {
            "model_heading": "Model {i}: {name}",
            "one_liner": "Summary", "evidence": "Evidence", "application": "Application", "limitation": "Limitation",
            "scenario": "Scenario", "case_example": "Case Example",
            "profile": "Profile", "core_conflict": "Core Conflict", "strategy": "Strategy",
            "frequency_label": "Frequency", "remedy": "Remedy",
            "title": "Dating Market Insight · Cognitive Operating System",
            "quote": '"The dating market has no feelings, only matching efficiency. Know your cards before you play them."',
            "usage_title": "How to Use",
            "usage_body": "This Skill is not a relationship counselor — it's a **dating market analysis tool**. It uses data and models to help you see:\n- Where you stand in the market\n- What your core conflict is\n- Which strategy is most likely to work\n\n**What it won't do**: no comfort, no platitudes, no \"you're great, you just haven't met the right person.\"",
            "workflow_title": "Analysis Workflow (Agentic Protocol)",
            "workflow_intro": "After receiving the user's personal profile, analyze as follows:",
            "step1_title": "Step 1: Information Gathering",
            "step1_body": "If the user hasn't provided complete info, ask for these key dimensions:\n- Gender, age, city\n- Occupation, annual income range\n- Education, height/weight\n- Self-assessed personality (3 keywords)\n- Top 3 partner requirements\n- Past relationship history (yes/no, how many)",
            "step2_title": "Step 2: Profile Mapping",
            "step2_body": "Map user info to the archetypes below, identify the closest 1-2 matches.",
            "step3_title": "Step 3: Market Valuation",
            "step3_body": "Assess market competitiveness across five dimensions: city tier, income, education, appearance, age.\nDon't score — spell out which cards are strong and which are weak.",
            "step4_title": "Step 4: Core Conflict Identification",
            "step4_body": "Use mental models to analyze structural contradictions between the user's profile and expectations.",
            "step5_title": "Step 5: Strategy Recommendation",
            "step5_body": "Based on heuristics, provide actionable matching strategy:\n- Where to look (social circle positioning)\n- What to emphasize (leverage strengths)\n- What to let go (calibrate expectations)\n- What to avoid (anti-pattern awareness)",
            "models_title": "Core Mental Models",
            "heuristics_title": "Decision Heuristics",
            "archetypes_title": "Archetypes",
            "anti_title": "Anti-Patterns (Common Mistakes)",
            "dna_title": "Expression DNA",
            "dna_intro": "Follow this style when analyzing:",
            "dna_style": "Style", "dna_sentence": "Sentence Pattern", "dna_vocab": "Key Vocabulary", "dna_tone": "Tone", "dna_taboo": "Taboo",
            "boundary_title": "Honest Boundaries",
            "boundary_items": [
                "Sample is biased toward Douyin (TikTok China) users aged 20-38, mostly in tier-1/2 cities — not representative of the entire dating market",
                "Data collected up to April 2026; subsequent market changes are not covered",
                "All models and heuristics are statistical patterns, not iron laws — individual variation always exists",
                "Does not cover relationship maintenance or post-marriage topics",
                "Not a substitute for professional psychological counseling",
            ],
            "sample_bias": "Female cases ({female}) outnumber male cases ({male}); male archetypes may be less comprehensive",
            "appendix_title": "Appendix: Data Sources",
            "appendix_platform": "Douyin (TikTok China)",
            "appendix_blogger": "Chen Nan (matchmaking analysis series)",
            "footer": "This Skill was auto-generated by the data distillation pipeline\n> Data source: Chen Nan matchmaking analysis series",
            "trigger_desc": 'Trigger when user mentions "matchmaking analysis", "dating market", "find a partner", "matching strategy", "market value".\n  Also trigger on "analyze my profile", "what kind of partner should I look for", "why can\'t I find someone".',
            "pattern_lib_title": "Conflict Pattern Library",
            "pattern_cases": "Representative cases",
            "archive_title": "Case Archive",
            "archive_headers": "| Case ID | Gender | Age | Income | Core Conflict |",
            "vocab_sep": ", ",
            # references i18n
            "stats_title": "Statistics Snapshot",
            "stats_generated": "Based on {total} cases",
            "stats_gender": "Gender", "stats_age": "Age", "stats_city": "City Tier",
            "stats_income": "Income Distribution", "stats_edu": "Education Distribution",
            "stats_tags_title": "Personality Tags Top {n}",
            "stats_male_tags": "Male Tags Top {n}", "stats_female_tags": "Female Tags Top {n}",
            "stats_cooccur": "Tag Co-occurrence Top {n}",
            "stats_conflict_kw": "Conflict Keywords Top {n}", "stats_strategy_kw": "Strategy Keywords Top {n}",
            "stats_pattern_dist": "Conflict Pattern Distribution",
            "pattern_typical": "Typical manifestations",
            "digest_title": "De-identified Case Digest",
            "digest_note": "De-identified summaries of all cases. Specific cities and employers have been generalized. Gender, age range, income tier, core conflict, and strategy essence are preserved.",
        },

        "qc": {
            "model_re": r'^###\s+Model\s*\d',
            "limit_re": r'Limitation|fail',
            "archetype_re": r'^###\s+[♂♀]',
            "boundary_section_re": r'##\s+Honest Boundaries(.*?)(?=\n##|\Z)',
            "anti_re": r'^\d+\.\s+\*\*.*?Frequency',
        },
    },
}
