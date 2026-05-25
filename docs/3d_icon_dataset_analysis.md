# 3D Icon 数据集分析报告

## 基本信息

| 项目 | 详情 |
|------|------|
| Hub URL | https://huggingface.co/datasets/linoyts/3d_icon |
| 样本总数 | 23 |
| 总大小 | ~25 MB |
| 图片来源 | [Unsplash](https://unsplash.com/) 免费授权图片 |
| 作者 | Maria Shalabaieva, Alexander Shatov |

---

## 存储格式

数据集使用 HuggingFace 的 **Image Folder** 格式（非 Parquet），Hub 上的文件组织为：

```
linoyts/3d_icon/
├── metadata.jsonl        # 元数据，每行一个 JSON 对象
├── 00.jpg ~ 22.jpg       # 23 张原始 JPEG 图片（独立文件，未压缩）
└── README.md             # 数据集说明卡
```

本地缓存路径：

```
~/.cache/huggingface/hub/datasets--linoyts--3d_icon/
├── blobs/                # Content-addressable 存储（文件名为 SHA256 hash）
│   ├── <hash>.jpg × 23   # 原始 JPEG 图片
│   ├── <hash>            # metadata.jsonl
│   └── <hash>            # README.md
├── refs/
│   └── main              # 指向当前 commit hash
└── snapshots/
    └── <commit_hash>/    # 版本快照（符号链接到 blobs）
```

---

## 数据字段

### metadata.jsonl 中的字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `file_name` | string | 原始 JPEG 文件名 | `"06.jpg"` |
| `prompt` | string | 图片文本描述，均以 `"a 3dicon, "` 开头 | `"a 3dicon, the tik tok logo is shown on a dark background"` |

### datasets 库加载后的字段

通过 `load_dataset('linoyts/3d_icon')` 加载后，每条数据包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | PIL.Image.Image (RGB) | 自动从 JPEG 解码的 PIL 图片对象 |
| `prompt` | str | 文本描述 |

---

## 原始 JSONL 示例

```json
{"file_name": "06.jpg", "prompt": "a 3dicon, the tik tok logo is shown on a dark background"}
{"file_name": "00.jpg", "prompt": "a 3dicon, a group of colorful icons on a black background"}
{"file_name": "21.jpg", "prompt": "a 3dicon, the app icon for the app is shown"}
{"file_name": "18.jpg", "prompt": "a 3dicon, a green owl icon on a green background"}
{"file_name": "01.jpg", "prompt": "a 3dicon, the icon for the messenger app on a colorful background"}
```

---

## 图片统计

| 分辨率 | 数量 | 占比 |
|--------|------|------|
| 3200 × 2400 | 9 | 39% |
| 4560 × 3600 | 6 | 26% |
| 3840 × 2160 | 4 | 17% |
| 4560 × 3100 | 1 | 4% |
| 4096 × 2784 | 1 | 4% |
| 12800 × 9600 | 3 | 13% |
| 8640 × 15360 | 1 | 4% (竖版) |

- 图片格式：JPEG (RGB)
- 分辨率范围：3200×2400 ~ 12800×9600
- 单张文件大小：~160 KB ~ 10.7 MB

---

## 全部 23 条数据

| # | file_name | prompt | 分辨率 |
|---|-----------|--------|--------|
| 0 | 06.jpg | a 3dicon, the tik tok logo is shown on a dark background | 3200×2400 |
| 1 | 00.jpg | a 3dicon, a group of colorful icons on a black background | 3200×2400 |
| 2 | 21.jpg | a 3dicon, the app icon for the app is shown | 4560×3600 |
| 3 | 18.jpg | a 3dicon, a green owl icon on a green background | 3840×2160 |
| 4 | 01.jpg | a 3dicon, the icon for the messenger app on a colorful background | 4560×3100 |
| 5 | 11.jpg | a 3dicon, twitter icon with blue background 3d illustration | 3200×2400 |
| 6 | 15.jpg | a 3dicon, a blue button with a white face on it | 3840×2160 |
| 7 | 22.jpg | a 3dicon, coinbase launches crypto lending service | 4560×3600 |
| 8 | 02.jpg | a 3dicon, a colorful phone with a rainbow background | 4096×2784 |
| 9 | 04.jpg | a 3dicon, uber logo on a black background with a clock and a phone | 4560×3600 |
| 10 | 09.jpg | a 3dicon, spotify app icon with music notes | 3200×2400 |
| 11 | 05.jpg | a 3dicon, snapchat icon with a ghost and other objects | 12800×9600 |
| 12 | 03.jpg | a 3dicon, lyft app icon | 4560×3600 |
| 13 | 14.jpg | a 3dicon, linkedin logo on blue background | 3200×2400 |
| 14 | 10.jpg | a 3dicon, instagram icon on pink background 3d render | 12800×9600 |
| 15 | 08.jpg | a 3dicon, facebook messenger and messenger logo | 12800×9600 |
| 16 | 19.jpg | a 3dicon, netflix logo with popcorn and a cup | 4560×3600 |
| 17 | 17.jpg | a 3dicon, a green and white logo on a blue background | 3840×2160 |
| 18 | 12.jpg | a 3dicon, netflix logo on a dark background | 3200×2400 |
| 19 | 20.jpg | a 3dicon, two black and pink boxes with the word uber on them | 4560×3600 |
| 20 | 16.jpg | a 3dicon, music note icon on a red background | 3840×2160 |
| 21 | 07.jpg | a 3dicon, a red play button on a dark blue background | 3200×2400 |
| 22 | 13.jpg | a 3dicon, a white and black square button with a black letter | 8640×15360 |

---

## 使用方式

```python
from datasets import load_dataset

ds = load_dataset('linoyts/3d_icon', split='train')

# 访问第一条数据
sample = ds[0]
image = sample['image']   # PIL Image
prompt = sample['prompt']  # str

# 保存图片
image.save('output.jpg')
```
