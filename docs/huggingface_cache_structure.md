# HuggingFace Hub 本地缓存结构详解

## 整体目录结构

通过 `datasets` 库或 `huggingface_hub` 下载的数据集/模型，缓存在 `~/.cache/huggingface/hub/` 下，以 `datasets--{org}--{name}` 或 `models--{org}--{name}` 命名：

```
~/.cache/huggingface/hub/datasets--linoyts--3d_icon/
├── blobs/           # 实际数据（内容寻址存储）
├── refs/            # 分支/tag 引用
├── snapshots/       # 版本快照（带原始文件名的符号链接）
└── .no_exist/       # 404 缓存
```

---

## 各目录详解

### `blobs/` — 数据仓库

存放所有实际文件内容，**以 SHA256 hash 命名**，不保留原始文件名和扩展名。

```
blobs/
├── 6158e94cf1e0c69a79703902081b6db72ccfddd2        # 实际是 metadata.jsonl
├── b732149eb581602abb6686728004f9c88505998b        # 实际是 README.md
├── 0a8a6fbdf3c0130def6011833db19967fc4c315d...    # 实际是 22.jpg
├── 4ef78806ff517a6f9664fd2161ce6d627709ae5c...    # 实际是 00.jpg
└── ...
```

**设计目的：**

| 特性 | 说明 |
|------|------|
| 去重 | 相同内容只存一份，即使文件名不同 |
| 完整性校验 | 文件名就是 SHA256 校验码，下载损坏立刻发现 |
| 版本共享 | 不同 commit 之间相同的文件不重复存储 |
| 格式透明 | 文件内容原样存储，不做任何二次压缩或封装 |

如何判断 blob 的实际格式：

```bash
# 看文件头部 magic bytes
xxd blobs/<hash> | head -1

# ff d8 ff → JPEG
# 89 50 4e 47 → PNG
# 50 41 52 31 → Parquet
# 7b → JSON
```

---

### `snapshots/` — 版本快照

每个 commit 一个子目录，里面是**符号链接**指向 `blobs/`，还原了原始文件名和目录结构：

```
snapshots/
└── 9574072d918bee2757215c9f2c8831a048001e27/    # commit hash
    ├── 00.jpg → ../../blobs/4ef78806...
    ├── 01.jpg → ../../blobs/d62ef8c1...
    ├── ...
    ├── 22.jpg → ../../blobs/0a8a6bfd...
    ├── metadata.jsonl → ../../blobs/6158e94c...
    └── README.md → ../../blobs/b732149e...
```

**这是用正常文件名访问数据的入口。**

如果数据集有多次更新（多个 commit），会有多个 snapshot 子目录，但相同的 blob 文件只存一份。

---

### `refs/` — 分支引用

记录分支名/tag 到 commit hash 的映射，类似 Git 的 `.git/refs/heads/`：

```
refs/
└── main    # 文件内容：9574072d918bee2757215c9f2c8831a048001e27
```

`load_dataset()` 默认读取 `refs/main` 指向的 commit 对应的 snapshot。

---

### `.no_exist/` — 404 缓存

记录哪些文件在某个 commit 下确认不存在，避免重复向服务器发无效请求：

```
.no_exist/
└── 9574072d918bee2757215c9f2c8831a048001e27/
    └── .gitattributes    # 表示该 commit 下没有此文件
```

作用：加速后续加载，减少无效网络请求。

---

## 与 Git 的类比

| 概念 | Git | HuggingFace Hub |
|------|-----|-----------------|
| 内容存储 | `.git/objects/` (SHA1) | `blobs/` (SHA256) |
| 版本视图 | working tree / checkout | `snapshots/<commit>/` (symlinks) |
| 分支指针 | `.git/refs/heads/` | `refs/` |
| 不存在标记 | 无 | `.no_exist/` |
| 大文件处理 | Git LFS | 直接存储在 blobs |
| 寻址方式 | SHA1 (40 char) | SHA256 (64 char) |

---

## 数据集加载时的查找路径

```python
load_dataset('linoyts/3d_icon', split='train')
```

内部查找顺序：

```
1. refs/main → 读取 commit hash: "9574072d..."
2. snapshots/9574072d.../ → 找到带原始文件名的 symlinks
3. 通过 symlinks → blobs/<hash> → 读取实际数据
4. 解析 metadata.jsonl → 建立 file_name 到 prompt 的映射
5. 构建 Arrow 缓存 → ~/.cache/huggingface/datasets/linoyts___3d_icon/
```

---

## 补充：`~/.cache/huggingface/datasets/` 目录

除了 `hub/` 之外，还有一个 `datasets/` 目录，存放 **Arrow 格式的处理缓存**：

```
~/.cache/huggingface/datasets/linoyts___3d_icon/
└── default/0.0.0/9574072d.../
    ├── dataset_info.json       # 数据集元信息
    └── train/
        ├── state.json          # 缓存状态
        └── *.arrow             # Arrow 列存格式（快速随机访问）
```

| 目录 | 存什么 | 用途 |
|------|--------|------|
| `hub/` | 原始下载文件 | 永久缓存，避免重复下载 |
| `datasets/` | Arrow 处理结果 | 加速后续 `load_dataset()` 调用 |

---

## 实用命令

```bash
# 查看缓存占用
du -sh ~/.cache/huggingface/hub/datasets--*

# 用正常文件名浏览数据集
ls ~/.cache/huggingface/hub/datasets--linoyts--3d_icon/snapshots/*/

# 清理特定数据集缓存
rm -rf ~/.cache/huggingface/hub/datasets--linoyts--3d_icon

# 清理所有缓存
huggingface-cli delete-cache

# 修改缓存路径（在下载前设置）
export HF_HOME=/your/custom/path
```
