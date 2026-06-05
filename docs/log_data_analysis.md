# 日志数据分析报告

## 基本信息

- **文件路径**: `/root/paddlejob/workspace/env/vfs_benchmark_cnn/zhuyinghao/260601_result`
- **文件大小**: 9.6GB
- **日期范围**: 2026-06-01
- **来源**: 百度 AI 助手图片生成服务 (`aichat/to_image`) 的 Agent 日志
- **总行数**: 1,878,138

## 三种日志类型

| 类型 | 全量条数 | 占比 | 说明 |
|------|----------|------|------|
| `intent_recognition` (意图识别) | 938,710 | 50.0% | 请求入口，记录用户原始输入 |
| `text_to_image_result` (文生图) | 535,589 | 28.5% | 纯文本生图结果 |
| `image_to_image_result` (图生图) | 374,716 | 19.9% | 图片+文本编辑结果 |

另有少量 `diagram_render` / `diagram_intent`（图表类）

---

## 1. 意图识别日志 (`intent_recognition.py`)

记录用户请求的原始输入信息：

| 字段 | 说明 | 示例 |
|------|------|------|
| `qid` | 查询ID | `8667204696490392029` |
| `session_id` | 会话ID | `10283629133351281701` |
| `query` | 用户原始请求文本 | `"帮我画 虾仁头像"` |
| `image_url` | 用户上传的图片URL列表 | `['https://aisearch...']` 或 `[]` |
| `messages` | 对话历史 | `[{'role':'user','content':...}]` |
| `request_intention` | 意图类型: **0=图生图, 2=文生图** | 图生图1,567 / 文生图935 |
| `browser_type` | 终端类型 | `wise`（移动端） |
| `userinfo` | 用户信息（地理位置、UA、IP等） | 含城市、省份、UA、uid |
| `req_from` | 请求来源 | `Assistant` |

---

## 2. 文生图结果日志 (`text_to_image_gen.py`)

### 字段说明

| 字段 | 说明 | 分布 |
|------|------|------|
| `category` | 分类 | 全部为 `image_aigc_1` |
| `model_type` | 模型类型 | 全部为 `miaotu`（妙图） |
| `model` | 模型名称 | 全部为 `文生图-妙图` |
| `task_type` | 任务类型 | `text_to_image` |
| `text` | 完整提示词 | 如 `"四张插画围绕古代孝故事..."` |
| `ratio` | 宽高比 | 1:1(876), 9:16(375), 16:9(102), 4:3(100), 3:4(25) |
| `image_num` | 生成图片数 | 4张(1,409), 3张(53), 2张(11), 1张(5) |
| `status` | 生成状态 | success(1,409), partial_success(64), fail(5) |
| `original_prompt_count` | 原始提示词数 | - |
| `valid_prompt_count` | 有效提示词数 | - |
| `filtered_count` | 过滤数量 | - |
| `anti_text` | 反作弊标记 | 全部 False |
| `is_end` | 是否结束 | - |
| `image_results` | 生成图片列表 | 每项含 `img_url`, `width`, `height` |
| `pe_prompt` | PE增强后的结构化提示词 | JSON，含 `refined_user_intent`, `type`, `prompts` 列表 |

### pe_prompt 结构示例

```json
{
  "refined_user_intent": "生成一张宋雨琦的脚部特写照片",
  "specify_image_type": "no",
  "type": "人像",
  "width_height_ratio": "no_ratio",
  "summary": "四张宋雨琦脚部特写的人像照片...",
  "prompts": [
    "生成一张写实风格，宋雨琦的脚部特写照片...",
    "生成一张艺术风格，宋雨琦的脚部特写照片...",
    "生成一张清新风格，宋雨琦的脚部特写照片...",
    "生成一张梦幻风格，宋雨琦的脚部特写照片..."
  ]
}
```

---

## 3. 图生图结果日志 (`image_to_image_gen.py`)

### 字段说明

| 字段 | 说明 | 分布 |
|------|------|------|
| `category` | 分类 | 全部为 `image_aigc_2` |
| `model_type` | 模型类型 | 全部为 `miaotu_pic` |
| `model` | 模型名称 | 图生图-妙图(1,905), 变清晰(163), 去水印(155) |
| `text` | 编辑提示词 | 部分为空 |
| `ratio` | 宽高比 | 1:1(93), 9:16(58), 16:9(16), 4:3(3), 3:4(3) |
| `image_num` | 生成图片数 | 1张(739), 4张(59), 0张(88，即失败), 2张(29) |
| `status` | 生成状态 | success(835), fail(88), partial_success(28) |
| `intent` | 编辑意图类型 | **8(1,107), 1(81), 2(78), 3(6)** |
| `anti_image` | 反作弊标记 | False(863), True(88) |
| `is_end` | 是否结束 | - |
| `image_results` | 生成图片列表 | 每项含 `img_url`, `width`, `height`, `model`, `ratio`, `intent` |
| `pe_prompt` | PE增强后的提示词 | JSON数组，含 `query`, `image_list`, `image_ratio` |

### pe_prompt 结构示例

```json
[
  {
    "query": "将图片中的人物改为站着手捧蜡烛蛋糕，微笑着眼睛看向前方，周围添加气球鲜花，配文：阿双哥生日快乐，光效为自然光",
    "image_list": ["https://aisearch.bj.bcebos.com/..."],
    "image_ratio": "1:1"
  }
]
```

---

## 关键发现

1. **文生图成功率 96.4%** (1,409/1,478)，图生图成功率 87.8% (835/951)，图生图失败率明显更高
2. **文生图默认生成4张图**（95%的请求），图生图默认生成1张
3. **图生图有3种子模型**: 妙图编辑(主)、变清晰、去水印
4. **图生图 intent=8 占绝大多数**(1,107/1,272 = 87%)，可能代表"通用图片编辑"
5. **用户主要通过移动端访问** (`browser_type=wise`)
6. **pe_prompt 字段是最有价值的**: 包含了经过 PE (Prompt Engineering) 增强后的结构化提示词，对训练数据构建非常有用
