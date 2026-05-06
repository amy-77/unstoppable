# 婚恋分析模型 API 使用说明

## 服务地址

```
http://172.27.96.38:8000
```

## 接口

### 健康检查

```bash
curl http://172.27.96.38:8000/health
```

返回：
```json
{"status": "ok", "model": "qwen3_4b_zh_skill_ep8"}
```

### 预测接口

```
POST http://172.27.96.38:8000/predict
Content-Type: application/json
```

#### 请求格式

```json
{
  "subject_profile": {
    "gender": "female",
    "age": 32,
    "birth_year": 1994,
    "location": {
      "city": "纽约",
      "tier": "overseas"
    },
    "education": "硕士（新闻/媒体方向）",
    "career": {
      "primary": "新闻媒体人",
      "tags": ["绿卡", "海归", "独立女性"]
    },
    "financials": {
      "annual_income_fixed": 450000,
      "assets_desc": "名下有房，有独立存款，家庭资产A8.5级别",
      "income_tier": "A8.5"
    },
    "physicals": {
      "height_cm": 165,
      "weight_kg": 56,
      "appearance_comment": "有健身习惯，有明显文身，整体风格偏'飒'",
      "style_tag": "modern_cool"
    },
    "family": {
      "background_desc": "从马来西亚移居美国，靠个人努力获得绿卡，背景深厚"
    }
  },
  "expectations": {
    "requirements_desc": "寻找文化同频、能接住梗的ABC或精英华人，消费观一致，渴望有传统根基的婚姻。"
  },
  "metadata": {
    "case_id": "test_001"
  }
}
```

#### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| subject_profile | 是 | 被分析人的基本信息 |
| subject_profile.gender | 是 | male / female |
| subject_profile.age | 是 | 年龄 |
| subject_profile.location | 否 | 城市和城市等级 |
| subject_profile.education | 否 | 学历 |
| subject_profile.career | 否 | 职业信息 |
| subject_profile.financials | 否 | 收入和资产 |
| subject_profile.physicals | 否 | 身高体重外貌描述 |
| subject_profile.family | 否 | 家庭背景 |
| expectations | 否 | 择偶要求描述 |
| metadata | 否 | 元信息，case_id 会在返回中回显 |

#### curl 示例

```bash
curl -X POST http://172.27.96.38:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject_profile": {
      "gender": "male",
      "age": 30,
      "location": {"city": "杭州", "tier": "new_first_tier"},
      "education": "本科（计算机）",
      "career": {"primary": "程序员", "tags": ["大厂", "高薪"]},
      "financials": {"annual_income_fixed": 600000, "assets_desc": "有房有车"},
      "physicals": {"height_cm": 175, "appearance_comment": "普通"}
    },
    "expectations": {
      "requirements_desc": "希望找温柔顾家的女生，最好有稳定工作"
    }
  }'
```

#### 返回格式

```json
{
  "case_id": "test_001",
  "output": "模型完整分析文本（含思考过程和结论）",
  "structured": {
    "psychology_and_traits": {
      "personality_tags": ["标签1", "标签2", "..."],
      "behavioral_logic": "核心行为逻辑总结"
    },
    "matchmaking_intelligence": {
      "core_conflict": "核心矛盾",
      "market_value_assessment": "市场估值",
      "expert_strategy": "专家策略建议",
      "target_portrait": "理想匹配对象画像",
      "logic_chain": ["推理步骤1", "推理步骤2", "..."]
    }
  }
}
```

## Python 调用示例

```python
import requests

resp = requests.post("http://172.27.96.38:8000/predict", json={
    "subject_profile": {
        "gender": "female",
        "age": 28,
        "education": "本科",
        "career": {"primary": "护士"},
        "financials": {"annual_income_fixed": 150000},
    },
    "expectations": {
        "requirements_desc": "想找有上进心、人品好的男生"
    }
})

result = resp.json()
print(result["structured"]["matchmaking_intelligence"]["target_portrait"])
```

## Demo Case（实际推理结果）

以下是模型对测试集第一条数据（鳌烨101）的真实推理结果：

**输入：** 32岁女性，纽约，硕士（新闻/媒体），绿卡，年收入45万，A8.5家庭资产，有文身，风格偏飒。寻找文化同频的ABC或精英华人。

**模型输出（target_portrait）：**

> 28-35岁、家族资产A7-A8、海外华人背景、金融/科技/律所从业者、文化消费能力强但家境审查相对宽松（如二代或新移民精英）、外形清爽、性格偏知性但不必太闷——本质是找'有钱有圈但精神世界她能填满'的类型，而非'传统+有深度'的双全人。

**标准答案（target_portrait）：**

> 首选留美多年或二代华人（ABC），具备极高文化同频度，能认同独立生活方式并包容文身史，经济上需门当户对以防被收割。

**LLM-as-Judge 评分（满分5分）：**

| 维度 | 得分 |
|------|------|
| conflict_insight（核心冲突识别） | 4/5 |
| strategy_direction（策略方向） | 3/5 |
| logic_depth（逻辑深度） | 5/5 |
| persona_read（人物心理解读） | 3/5 |
| actionability（可执行性） | 5/5 |
| **综合均分** | **4.0/5** |

完整推理日志见 `test_inference_result.txt`。

## 注意事项

- 单次推理约需 10-15 分钟（GPU 资源紧张时），请设置足够的超时时间（建议 timeout=600）
- 服务运行在 `172.27.96.38`（qwang 的开发机），需在同一内网下访问
- 如果服务挂了，在服务器上执行：
  ```bash
  cd /data/qwang/q/thalia/agentic_hackson/agent/matchmaker/training
  CUDA_VISIBLE_DEVICES=0 nohup python3 -u serve.py --port 8000 > serve.log 2>&1 &
  ```
