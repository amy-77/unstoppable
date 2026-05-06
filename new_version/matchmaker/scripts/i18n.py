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
2. 心智模型必须包含至少2个关于内在人格结构与策略方向关系的模型(如: 什么类型的内在特质应该匹配什么类型的伴侣), 不能全部是外部条件匹配类模型. 特别要求提炼: 当用户自身特质具有高波动性时, 策略应指向互补而非同频的决策逻辑.
3. 心智模型必须包含至少1个关于策略节奏判断的模型——即什么条件下应该建议求偶者暂停入场、先做个人增值或心理建设, 而非立即匹配. 模型需明确暂停的触发条件(如: 自卑导致反复丧失底线、情伤未愈导致防御性拒绝、硬件潜力远未兑现)和恢复入场的标准(如: 连续3个月情绪稳定、硬件兑现到位、自我价值感恢复到能拒绝不合适对象的程度). 决策变量包括: 心理创伤修复程度、硬件兑现周期、窗口关闭速度、当前情绪稳定性. 该模型还必须覆盖「收牌vs打牌」的核心判断: 当求偶者结构性劣势过重(3项及以上弱势叠加)时, 应识别出「暂停入场、专注增值」是最优策略, 而非默认给出市场匹配方案.
4. 心智模型必须包含至少1个关于博弈姿态切换的模型——区分三种策略姿态及其切换条件: (a)进攻型: 主动出击扩大匹配池, 触发条件为自身条件达标且心理状态稳定; (b)防御型: 收缩战线防止被收割, 触发条件为弱势叠加或心理脆弱期; (c)锁定型: 已获超额机会时全力配合锁定成果, 触发条件为对方条件远超自身市场定位. 特别强调: 当价值严重不对称(对方远超自身)时, 策略应从「平等筛选」切换为「降低姿态、主动示弱、加速关系确认」, 此时「不卑微」「保持自我」等通用建议反而是最大风险.
5. 心智模型必须包含至少1个关于人格脆弱性如何影响市场风险的模型——不是所有失败都是市场结构问题, 有些是人格特质导致的被收割风险. 识别哪些人格组合(如: 单纯+高资产+低社会经验、自卑敏感+低议价力+讨好型人格、强控制+低情感细腻度+高物质条件)会系统性地导致被错误对象吸引或被中介/平台反复收割, 以及对应的防御型策略(收缩社交半径、设置硬性筛选门槛、引入第三方评估机制).
6. 心智模型必须包含至少1个关于行动时机判断的模型——即: 什么条件下建议暂停/沉淀而非立即行动, 暂停的时间尺度如何确定(短暂暂停3-6个月 vs 长期沉淀1-3年), 重新入场的触发条件是什么. 决策变量包括: 心理创伤修复程度、硬件兑现周期、窗口关闭速度、当前情绪稳定性. 当用户存在未愈合的情感创伤或硬件潜力远未兑现时, 默认建议暂停并给出明确的重新入场触发条件, 而非强行推动行动.
7. 心智模型必须包含至少1个关于经济博弈/财富分配的模型——识别高资产差异匹配中的结构性因素: 控制权归属(谁掌握经济主导权)、家族审查机制(对方家庭对资产保护的介入程度)、婚前保护策略(协议/信托/财产隔离), 以及什么信号表明匹配本质是「资产整合」而非「感情结合」.
8. 心智模型必须包含至少1个关于策略粒度的模型——即什么情况下应输出具体行动指令(联系谁、去哪、做什么)而非仅输出方向性建议. 判断变量包括: 用户当前是否有明确可触达的目标对象、用户执行力水平、窗口紧迫度.
9. 心智模型必须包含至少1个关于市场风险/剥削机制的模型——识别什么条件组合下用户不是在「匹配」而是在「被收割」, 以及对应的防御型策略方向.
10. 心智模型必须包含1个关于入场时机判断的模型——区分何时应趁窗口期抢先布局 vs 何时应暂停等兑现后再入场, 给出明确的决策变量(当前议价力/潜力兑现周期/窗口关闭速度).
11. 心智模型必须包含至少1个关于博弈态势判断的模型——区分三种策略姿态: 进攻型(主动出击扩大匹配池)、防御型(收缩战线防止被收割)、锁定型(已获超额机会时全力配合锁定成果). 明确每种姿态的触发条件和切换信号.
12. 心智模型必须包含至少1个关于内因vs外因归因优先级的模型——当用户硬件条件达标(收入/学历/城市均不低于目标池中位数)但转化率持续低迷时, 内因(恋人属性缺失、情感诱导力不足、互动模式僵化)优先于外部结构(地域壁垒、池子窄、渠道问题)作为核心归因. 给出明确的判断标准: 硬件达标的定义、转化率低迷的阈值.
13. 每个心智模型必须包含「common_misjudgment」字段——说明该模型容易被什么表面现象干扰导致误判.
14. 对每个心智模型, 必须标注其策略方向倾向(进攻型: 积极入场匹配 / 防御型: 暂停修复再入场 / 锁定型: 已有目标全力锁死), 并说明触发条件. 避免所有模型都指向进攻型策略, 至少2个模型的策略方向倾向为防御型或锁定型.
15. 决策启发式8-12条, 要具体到可执行
16. 启发式规则必须包含至少2条关于策略方向判断的规则(什么情况下建议暂停vs立即行动、什么情况下建议互补vs同频、什么情况下问题在内部vs外部), 这些规则必须有明确的判断条件和边界, 不能是模糊的原则.
17. 启发式规则必须包含至少2条防御性规则(什么情况下不要做某事), 而非全部是进攻性规则(应该做某事). 每条防御性规则必须有明确的触发条件和替代行动.
18. 启发式规则必须包含关于分析基调/措辞烈度的规则——什么条件下应使用温和建议型措辞, 什么条件下必须切换为尖锐警示型措辞. 给出明确的触发条件.
19. 启发式规则必须包含至少1条关于特殊资产(绿卡、户口、家族企业、稀缺编制等)的双面性判断规则.
20. 所有内容必须有案例数据支撑, 不要编造
21. 每条证据和案例佐证要简短(一句话)

特别关注以下反直觉模式的提炼：
(1) 条件越好越不该主动出击的场景——当用户已获得超额匹配机会（对方条件远超自身市场定位）时，正确策略是全力配合锁定而非继续筛选博弈；
(2) 应该全力配合而非保持独立的场景——当博弈态势为「锁定型」时，「不卑微」「保持自我」等通用建议反而是最大风险，正确策略是主动示弱、降低对方防御、加速关系确认；
(3) 核心问题不在外部环境而在自身互动模式的场景——硬件达标但持续失败时，「换城市」「换渠道」「扩圈子」都是逃避，真正需要改变的是情感表达能力和互动模式。
这些模式往往与通用建议相反，但在特定博弈态势下是最优解，必须在模型和启发式中体现。""",

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
      "profile": "画像描述（一句话，以人格内核为主轴：依附类型/核心恐惧/情感模式，职业仅作辅助标签）",
      "core_conflict": "核心矛盾（一句话）",
      "strategic_direction": "方向性策略（一句话，必须包含时间维度：现在就做/先等3-6个月/长期准备1-3年，以及方向）",
      "strategy_tempo": "该原型的默认策略节奏（immediate_action / short_pause_3to6m / long_pause_1to3y）及判断依据（一句话）",
      "immediate_action": "72小时内可执行的具体动作示例（一句话，如：本周内联系某类人/去某类场所/做某个具体行为。若strategy_tempo为暂停型，则此处写暂停期内的自我建设动作）"
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
2. 原型的profile描述必须以人格内核（依附类型/核心恐惧/情感模式/防御机制）为第一描述维度，职业和经济条件仅作辅助标签。避免将职业身份作为性格解释的主因。例如：不要写「因为是程序员所以不善表达」，而要写「回避型依附+情感表达抑制，职业为技术岗」。
2.5. 原型的strategic_direction必须区分三种情况并确保覆盖: (1)立即入场型——给匹配策略和具体行动(至少覆盖3个原型); (2)暂停修复型——给增值/心理建设方案, strategy_tempo为short_pause_3to6m或long_pause_1to3y(至少覆盖2个原型); (3)方向翻转型——给与直觉相反的建议, 如条件好但应该降维匹配、条件差但可以向上突破(至少覆盖2个原型). 每种类型的判断依据必须写入strategic_direction字段.
2.6. 每个原型的strategic_direction必须明确标注归因重心(外部市场结构 / 内在人格改造 / 混合型), 且至少有2个原型的归因重心为「内在人格改造」(如: 需要学会情感表达、需要建立情绪吸引力、需要修复心理创伤后再入场). 避免所有原型的策略都指向外部市场操作(换渠道、调标准、降维).
3. 每个原型必须标注其核心策略方向是同频匹配还是互补匹配，并说明判断依据。在strategic_direction字段中明确写出「策略姿态：X，策略方向：同频/互补，节奏：现在就做/暂停N个月/长期准备，依据：...」。原型的strategic_direction必须明确包含时间维度——不仅说明方向，还要说明节奏（现在就做 vs 先等再做 vs 长期准备）。在immediate_action字段中给出该原型72小时内应执行的一个具体动作，不能仅停留在方向层面。每个原型的策略必须包含至少一个具体可执行动作。
4. 原型必须覆盖至少1个处于身份未兑现期的类型（在读博士/创业早期/职业转型期），标注其入场时机的策略分支：当前议价力为零时应暂停入场等兑现，还是应利用学历/潜力光环提前锁定目标。strategy_tempo应为short_pause_3to6m或long_pause_1to3y。
5. 原型必须覆盖至少1个「情伤未愈的高硬件个体」——硬件条件强但因近期情感创伤（被分手/被背叛/丧偶）导致心理防御封闭，strategy_tempo应为short_pause_3to6m，重新入场触发条件为情绪稳定性恢复指标。
6. 原型必须覆盖至少1个「暂停型高价值个体」——条件达标但心理/生活状态不支持有效行动（如刚经历重大变故、正在高强度职业转型、情绪能量极低），strategy_tempo应为long_pause_1to3y。
7. 原型必须覆盖以下维度组合：
   (1) 硬件强+软件弱型——硬件条件顶配（高收入/高学历/高颜值）但恋人属性为零（不会聊天、不懂情绪价值、互动模式僵化），策略姿态应为「暂停+内修」而非「进攻扩圈」，strategy_tempo为short_pause_3to6m；
   (2) 已获超额机会但可能丢失型——已匹配到远超自身市场定位的对象，但因心理障碍（不安全感、过度博弈、无法放下姿态）可能丢失机会，策略姿态应为「锁定」，strategy_tempo为immediate_action；
   (3) 高价值但高风险暴露型——被宠溺长大+高颜值/高资产+社会经验极浅，核心风险不是「找不到人」而是「被错误的人找到」，策略姿态应为「防御」，strategy_tempo为immediate_action（但行动内容是防御性的）；
   (4) 海外独立女性型——独立身份vs传统婚恋诉求的信号错位，文身/生活方式等作为隐性筛选器，需要明确信号管理策略。
8. 反模式5-8个，来自案例中反复出现的错误
9. 反模式部分必须包含至少1个关于策略方向误判的反模式（如：对高波动型用户建议同频匹配导致关系不稳定、对需要互补托底的用户错误建议找同类型伴侣）
10. 反模式必须包含至少1个关于分析系统自身策略方向错误的反模式——如: 对弱势叠加用户过度乐观定位(明明是被收割对象却建议「扩大社交圈」)、对吸引力缺失者只建议换渠道而不建议自我改造(外形管理/情绪价值输出能力提升)、对信息模糊案例给出条件分支而非确定性判断(该果断推断时却保守观望).
10.5. 反模式必须包含至少1个关于系统性策略偏差的反模式——如: 总是建议积极入场而忽视暂停选项(默认行动偏差)、总是归因于外部市场结构而忽视内在人格改造需求(外因优先偏差)、总是建议改变自己适配市场而忽视用真实自我筛选对的人(过度适配偏差). 该反模式的remedy必须包含具体的自检机制: 每次输出策略前检查是否考虑了暂停选项、是否考虑了内因归因、是否考虑了「不改变自己」的可能性.
11. 反模式必须包含至少1个关于分析者自身认知偏差的反模式——如：结构归因谬误（将所有问题归因于外部结构而忽视个体能力缺陷）、政治正确偏差（因避免「冒犯」而给出「平等」「不卑微」「双向评估」等安全建议，实际上在锁定型博弈中这些建议会导致机会丧失）、外因优先偏差（当硬件达标但转化率低时，仍优先建议换渠道/换城市而非改变互动模式）。
12. 反模式必须包含至少1个关于「默认行动偏差」的反模式——所有原型都被给出立即行动建议，而忽视部分原型（情伤未愈型/身份未兑现型/能量耗竭型）的最优策略是暂停。识别信号：用户描述中出现「刚分手」「还没走出来」「现在没心思」「先把事业搞好」等表述时，系统仍强推行动方案。
13. 表达DNA基于陈楠的分析风格：直接犀利、数据驱动、不留情面、善用类比
14. 所有描述尽量简短，一句话为主""",

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
