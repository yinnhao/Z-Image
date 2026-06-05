"""
从两个日志文件中分别随机抽取5万条 intent=8 的图生图数据，
去重后合并为10万条训练集。

去重规则: (prompt, source_image_urls, target_image_url) 完全相同则视为重复，只保留一条

用法:
    python build_train_dataset.py
"""

import ast
import hashlib
import json
import os
import random
import re

LOG_FILES = [
    "/root/paddlejob/workspace/env/vfs_benchmark_cnn/zhuyinghao/260531_result",
    "/root/paddlejob/workspace/env/vfs_benchmark_cnn/zhuyinghao/260601_result",
]
OUTPUT_DIR = "/root/Z-Image/output/edit_data_from_log"
DATASET_FILE = os.path.join(OUTPUT_DIR, "edit_train_100k.jsonl")

SEED = 42
SAMPLES_PER_FILE = 50000


def remove_watermark(url):
    """去掉百度图片链接中的水印参数 &wm=...&wmo=..."""
    return re.sub(r'&wm=[^?]*', '', url)


def parse_log_line(line):
    match = re.search(r"image_to_image_result\[(\{.*\})\]\s+message\[", line)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None


def extract_task_id(line):
    match = re.search(r"task_id\[([^\]]+)\]", line)
    return match.group(1) if match else None


def extract_edit_records(data, task_id):
    if data.get("status") != "success" or data.get("model_type") != "miaotu_pic":
        return []

    image_results = data.get("image_results", [])
    intent8_results = [r for r in image_results if isinstance(r, dict) and r.get("intent") == "8"]
    if not intent8_results:
        return []

    pe_prompt_str = data.get("pe_prompt", "")
    if not pe_prompt_str:
        return []
    try:
        pe_prompts = json.loads(pe_prompt_str)
    except json.JSONDecodeError:
        return []
    if not isinstance(pe_prompts, list) or not pe_prompts:
        return []

    records = []
    for i, pe in enumerate(pe_prompts):
        if i >= len(intent8_results):
            break
        query = pe.get("query", "").strip()
        source_images = pe.get("image_list", [])
        image_ratio = pe.get("image_ratio", "")
        if not query or not source_images:
            continue
        result = intent8_results[i]
        target_url = result.get("img_url", "")
        if not target_url:
            continue
        records.append({
            "task_id": task_id,
            "prompt": query,
            "source_image_urls": [remove_watermark(u) for u in source_images],
            "source_image_count": len(source_images),
            "target_image_url": remove_watermark(target_url),
            "target_width": result.get("width", 0),
            "target_height": result.get("height", 0),
            "ratio": image_ratio or result.get("ratio", ""),
        })
    return records


def dedup_key(rec):
    """生成去重键: prompt + 排序后的源图URL（结果图一定不同，不需参与去重）"""
    src_urls = tuple(sorted(rec["source_image_urls"]))
    return (rec["prompt"], src_urls)


def extract_from_file(log_file, max_samples):
    """从单个日志文件提取并随机抽取指定数量的记录。"""
    print(f"\n{'='*60}")
    print(f"处理文件: {log_file}")
    print(f"目标样本数: {max_samples}")
    print(f"{'='*60}")

    all_records = []
    seen_keys = set()
    skipped_parse = 0
    skipped_no_intent8 = 0
    skipped_dup = 0
    processed_lines = 0

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if "image_to_image_result" not in line:
                continue
            processed_lines += 1
            task_id = extract_task_id(line)
            data = parse_log_line(line)
            if data is None:
                skipped_parse += 1
                continue
            records = extract_edit_records(data, task_id)
            if not records:
                skipped_no_intent8 += 1
                continue

            for rec in records:
                key = dedup_key(rec)
                if key in seen_keys:
                    skipped_dup += 1
                    continue
                seen_keys.add(key)
                all_records.append(rec)

            if processed_lines % 100000 == 0:
                print(f"  已扫描 {processed_lines} 行, 累积去重记录 {len(all_records)} 条")

    print(f"  扫描完成: {processed_lines} 行")
    print(f"  解析失败: {skipped_parse}")
    print(f"  非intent8/非success: {skipped_no_intent8}")
    print(f"  同文件内重复: {skipped_dup}")
    print(f"  去重后总记录: {len(all_records)}")

    # 随机抽取
    if len(all_records) <= max_samples:
        print(f"  记录数({len(all_records)}) <= 目标({max_samples}), 全部保留")
        return all_records
    else:
        random.seed(SEED)
        sampled = random.sample(all_records, max_samples)
        print(f"  随机抽取 {max_samples} 条")
        return sampled


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    all_sampled = []

    for log_file in LOG_FILES:
        records = extract_from_file(log_file, SAMPLES_PER_FILE)
        all_sampled.extend(records)

    # 跨文件去重
    print(f"\n{'='*60}")
    print(f"跨文件去重...")
    seen_keys = set()
    final_records = []
    cross_dup = 0
    for rec in all_sampled:
        key = dedup_key(rec)
        if key in seen_keys:
            cross_dup += 1
            continue
        seen_keys.add(key)
        final_records.append(rec)

    # 打乱最终数据
    random.shuffle(final_records)

    print(f"  合并总数: {len(all_sampled)}")
    print(f"  跨文件重复: {cross_dup}")
    print(f"  最终数据集: {len(final_records)} 条")

    # 写入 JSONL
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n数据集已保存: {DATASET_FILE}")

    # 统计信息
    src_counts = {}
    for r in final_records:
        c = r.get("source_image_count", 1)
        src_counts[c] = src_counts.get(c, 0) + 1
    print(f"\n源图数量分布:")
    for k in sorted(src_counts):
        print(f"  {k}张源图: {src_counts[k]}条")

    # 生成可视化 (取前100条)
    generate_visualization(final_records[:100])


def generate_visualization(records):
    """从记录生成可视化 HTML 页面（前100条预览）。"""
    multi_img_count = sum(1 for r in records if r.get("source_image_count", 1) > 1)

    cards_html = []
    for i, rec in enumerate(records):
        prompt = rec["prompt"]
        ratio = rec.get("ratio", "")
        size_info = f"{rec.get('target_width', '?')}x{rec.get('target_height', '?')}"
        src_urls = rec.get("source_image_urls", [])
        src_count = rec.get("source_image_count", 1)
        tgt_url = rec["target_image_url"]

        src_imgs_html = ""
        for j, url in enumerate(src_urls):
            url_esc = url.replace("&", "&amp;").replace('"', "&quot;")
            label = f"Source {j+1}" if src_count > 1 else "Source (输入)"
            src_imgs_html += f'<img src="{url_esc}" alt="source{j}" loading="lazy" onclick="openModal(this.src)" title="{label}">\n'

        tgt_url_esc = tgt_url.replace("&", "&amp;").replace('"', "&quot;")
        src_count_badge = f' | <span style="color:#e67e22">源图x{src_count}</span>' if src_count > 1 else ""

        card = f"""
        <div class="card">
            <div class="card-header">
                <span class="card-index">#{i + 1}</span>
                <span class="card-meta">ratio: {ratio} | size: {size_info}{src_count_badge}</span>
            </div>
            <div class="card-prompt">{prompt}</div>
            <div class="card-images">
                <div class="image-box source-box">
                    <div class="image-label">Source (输入) {f'×{src_count}' if src_count > 1 else ''}</div>
                    <div class="source-grid {'multi' if src_count > 1 else ''}">
                        {src_imgs_html}
                    </div>
                </div>
                <div class="arrow">&#10132;</div>
                <div class="image-box">
                    <div class="image-label">Target (输出)</div>
                    <img src="{tgt_url_esc}" alt="target" loading="lazy" onclick="openModal(this.src)">
                </div>
            </div>
        </div>"""
        cards_html.append(card)

    viz_file = os.path.join(OUTPUT_DIR, "visualization_100k.html")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>图生图训练数据预览 (前100条)</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
h1 {{ text-align: center; color: #333; margin-bottom: 10px; font-size: 24px; }}
.summary {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }}
.card {{ background: #fff; border-radius: 12px; margin-bottom: 20px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.card-index {{ background: #4a90d9; color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: bold; }}
.card-meta {{ color: #999; font-size: 12px; }}
.card-prompt {{ background: #f0f4ff; border-left: 4px solid #4a90d9; padding: 10px 14px; border-radius: 4px; margin-bottom: 14px; font-size: 14px; line-height: 1.6; color: #333; word-break: break-all; }}
.card-images {{ display: flex; align-items: center; gap: 16px; }}
.image-box {{ flex: 1; text-align: center; }}
.source-box {{ flex: 2; }}
.image-label {{ font-size: 12px; color: #888; margin-bottom: 6px; font-weight: 500; }}
.image-box img {{ max-width: 100%; max-height: 400px; border-radius: 8px; border: 1px solid #e0e0e0; cursor: pointer; transition: transform 0.2s; object-fit: contain; }}
.image-box img:hover {{ transform: scale(1.02); }}
.source-grid {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }}
.source-grid.multi img {{ max-height: 180px; max-width: 48%; }}
.arrow {{ font-size: 28px; color: #4a90d9; font-weight: bold; flex-shrink: 0; }}
.modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 1000; justify-content: center; align-items: center; cursor: pointer; }}
.modal.active {{ display: flex; }}
.modal img {{ max-width: 90%; max-height: 90%; border-radius: 8px; }}
.stats {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 20px; }}
.stat-item {{ text-align: center; }}
.stat-num {{ font-size: 28px; font-weight: bold; color: #4a90d9; }}
.stat-label {{ font-size: 12px; color: #999; }}
</style>
</head>
<body>
<h1>图生图训练数据预览</h1>
<p class="summary">从前100条数据预览 | 点击图片可放大查看</p>
<div class="stats">
    <div class="stat-item"><div class="stat-num">{len(records)}</div><div class="stat-label">本页预览</div></div>
    <div class="stat-item"><div class="stat-num">{multi_img_count}</div><div class="stat-label">多源图样本</div></div>
</div>
{''.join(cards_html)}
<div class="modal" id="modal" onclick="closeModal()">
    <img id="modal-img" src="">
</div>
<script>
function openModal(src) {{
    document.getElementById('modal-img').src = src;
    document.getElementById('modal').classList.add('active');
}}
function closeModal() {{
    document.getElementById('modal').classList.remove('active');
}}
document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') closeModal();
}});
</script>
</body>
</html>"""

    with open(viz_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"预览页面已生成: {viz_file}")


if __name__ == "__main__":
    main()
