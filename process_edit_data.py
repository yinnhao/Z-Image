"""
从日志中提取 intent=8 的图生图数据，生成可视化 HTML 页面。

用法:
    python process_edit_data.py              # 默认100条
    python process_edit_data.py --max_samples 500
"""

import ast
import argparse
import json
import os
import re


LOG_FILE = "/root/paddlejob/workspace/env/vfs_benchmark_cnn/zhuyinghao/260601_result"
OUTPUT_DIR = "/root/Z-Image/output/edit_data_from_log"
DATASET_FILE = os.path.join(OUTPUT_DIR, "edit_dataset.jsonl")
VIZ_FILE = os.path.join(OUTPUT_DIR, "visualization.html")


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


def process_logs(max_samples=100):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    processed_lines = 0
    skipped = 0

    print(f"开始处理日志: {LOG_FILE}")
    print(f"目标样本数: {max_samples}")
    print("-" * 60)

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "image_to_image_result" not in line:
                continue
            processed_lines += 1
            task_id = extract_task_id(line)
            data = parse_log_line(line)
            if data is None:
                skipped += 1
                continue
            records = extract_edit_records(data, task_id)
            if not records:
                skipped += 1
                continue
            for rec in records:
                all_records.append(rec)
                if len(all_records) >= max_samples:
                    break
            if len(all_records) >= max_samples:
                break

    # Write JSONL
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"扫描 image_to_image_result 行数: {processed_lines}")
    print(f"跳过(解析失败/非intent8): {skipped}")
    print(f"有效样本数: {len(all_records)}")
    print(f"数据集文件: {DATASET_FILE}")
    return all_records


def generate_visualization(records):
    # 统计多图信息
    multi_img_count = sum(1 for r in records if r.get("source_image_count", 1) > 1)

    cards_html = []
    for i, rec in enumerate(records):
        prompt = rec["prompt"]
        ratio = rec.get("ratio", "")
        size_info = f"{rec.get('target_width', '?')}x{rec.get('target_height', '?')}"
        src_urls = rec.get("source_image_urls", [])
        src_count = rec.get("source_image_count", 1)
        tgt_url = rec["target_image_url"]

        # Build source images HTML
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

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>图生图训练数据可视化 (intent=8)</title>
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
.filters {{ text-align: center; margin-bottom: 20px; }}
.filters button {{ padding: 6px 16px; margin: 0 4px; border: 1px solid #ddd; background: #fff; border-radius: 6px; cursor: pointer; font-size: 13px; }}
.filters button:hover {{ background: #e8e8e8; }}
.filters button.active {{ background: #4a90d9; color: #fff; border-color: #4a90d9; }}
.stats {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 20px; }}
.stat-item {{ text-align: center; }}
.stat-num {{ font-size: 28px; font-weight: bold; color: #4a90d9; }}
.stat-label {{ font-size: 12px; color: #999; }}
</style>
</head>
<body>
<h1>图生图训练数据可视化</h1>
<p class="summary">intent=8 | 点击图片可放大查看</p>
<div class="stats">
    <div class="stat-item"><div class="stat-num">{len(records)}</div><div class="stat-label">总样本数</div></div>
    <div class="stat-item"><div class="stat-num">{multi_img_count}</div><div class="stat-label">多源图样本</div></div>
</div>
<div class="filters">
    <button onclick="changeLayout('side')" class="active" id="btn-side">左右对比</button>
    <button onclick="changeLayout('grid')" id="btn-grid">上下对比</button>
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
function changeLayout(mode) {{
    document.getElementById('btn-side').classList.remove('active');
    document.getElementById('btn-grid').classList.remove('active');
    if (mode === 'side') {{
        document.getElementById('btn-side').classList.add('active');
        document.querySelectorAll('.card-images').forEach(el => el.style.flexDirection = 'row');
        document.querySelectorAll('.arrow').forEach(el => el.innerHTML = '&#10132;');
    }} else {{
        document.getElementById('btn-grid').classList.add('active');
        document.querySelectorAll('.card-images').forEach(el => el.style.flexDirection = 'column');
        document.querySelectorAll('.arrow').forEach(el => el.innerHTML = '&#11015;');
    }}
}}
document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') closeModal();
}});
</script>
</body>
</html>"""

    with open(VIZ_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"可视化页面已生成: {VIZ_FILE}")


def main():
    parser = argparse.ArgumentParser(description="从日志提取图生图训练数据")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()
    records = process_logs(max_samples=args.max_samples)
    generate_visualization(records)


if __name__ == "__main__":
    main()
