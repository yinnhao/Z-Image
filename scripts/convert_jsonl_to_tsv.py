#!/usr/bin/env python3
"""
将 JSONL 格式的编辑数据转换为 TSV 格式（兼容现有训练流程）
边处理边写入模式：每处理完一条立即写入，可随时看到进度

TSV 格式（8 字段，tab 分隔）：
    field 0: md5 ID
    field 1: 数据集标签
    field 2: JSON metadata
    field 3: 源图 base64 JPEG
    field 4: 目标图 base64 JPEG
    field 5: 短 prompt（中文）
    field 6: 中等 prompt（中文）
    field 7: 详细 prompt（中文）
"""

import base64
import hashlib
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import urllib3
from PIL import Image
from tqdm import tqdm

# 禁用 SSL 警告
urllib3.disable_warnings()

# HTTP 代理配置
PROXY = os.environ.get("https_proxy", "http://agent.baidu.com:8891")


# 全局锁，保证写入安全
write_lock = Lock()

def fix_url(url):
    """修复 URL，将 gips[0-3].baidu.com 替换为 yawen-gips.baidu-int.com"""
    if url and 'gips' in url:
        # gips0.baidu.com, gips1.baidu.com, gips2.baidu.com, gips3.baidu.com -> yawen-gips.baidu-int.com
        url = re.sub(r'gips[0-3]\.baidu\.com', 'yawen-gips.baidu-int.com', url)
    return url


def should_use_proxy(url):
    """内网域名和特定域名不使用代理"""
    # 这些域名不需要代理，直接访问
    no_proxy_domains = [
        'baidu-int.com',
        'aisearch.bj.bcebos.com',
        'aisearch.cdn.bcebos.com',
    ]
    for domain in no_proxy_domains:
        if url and domain in url:
            return False
    return True


def download_image(url, timeout=30, max_retries=3):
    """下载图片，返回 PIL Image 对象"""
    # 修复 URL
    url = fix_url(url)

    # 判断是否需要代理
    if should_use_proxy(url):
        http = urllib3.ProxyManager(
            PROXY,
            cert_reqs='CERT_NONE',
            timeout=urllib3.Timeout(connect=timeout, read=timeout),
            maxsize=10,
            retries=urllib3.Retry(total=max_retries, backoff_factor=0.5),
        )
    else:
        # 内网域名，不使用代理
        http = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=timeout, read=timeout),
            maxsize=10,
            retries=urllib3.Retry(total=max_retries, backoff_factor=0.5),
        )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/*",
    }

    try:
        resp = http.request("GET", url, headers=headers, preload_content=False)
        if resp.status != 200:
            raise Exception(f"HTTP {resp.status}")

        # 读取完整数据
        data = resp.read()
        return Image.open(io.BytesIO(data))
    finally:
        if 'resp' in locals():
            resp.release_conn()
        http.clear()


def image_to_base64(img: Image.Image, quality=85, max_size=1536) -> str:
    """将 PIL Image 转换为 base64 JPEG 字符串"""
    # 调整大小（如果需要）
    if max(img.width, img.height) > max_size:
        ratio = max_size / max(img.width, img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    # 确保是 RGB 模式
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 保存为 JPEG 字节流
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    img_bytes = buffer.getvalue()

    # base64 编码
    return base64.b64encode(img_bytes).decode('utf-8')


def generate_md5(content: str) -> str:
    """生成内容的 MD5 hash"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def create_metadata(data: dict) -> dict:
    """创建 JSON metadata 字段"""
    metadata = {
        "text": data.get("prompt", ""),
        "open_image_input_url": data.get("source_image_urls", [""])[0] if data.get("source_image_urls") else "",
        "output_image": data.get("target_image_url", ""),
        "edit_type": "image_edit",
        "md5": data.get("task_id", ""),
    }
    return metadata


def process_one_line(line: str, dataset_label: str = "online_data_edit") -> dict:
    """
    处理单条 JSONL 数据，返回 TSV 行字典
    """
    try:
        data = json.loads(line.strip())
        prompt = data.get("prompt", "")
        if not prompt:
            return {"success": False, "error": "empty prompt"}

        source_urls = data.get("source_image_urls", [])
        target_url = data.get("target_image_url", "")

        if not source_urls:
            return {"success": False, "error": "no source image"}
        if not target_url:
            return {"success": False, "error": "no target image"}

        # 只取第一个源图
        source_url = source_urls[0]

        # 下载图片
        try:
            source_img = download_image(source_url)
            target_img = download_image(target_url)
        except Exception as e:
            return {"success": False, "error": f"download failed: {e}"}

        # 转换为 base64
        source_b64 = image_to_base64(source_img, quality=85, max_size=1536)
        target_b64 = image_to_base64(target_img, quality=85, max_size=1536)

        # 生成 md5 ID
        md5_id = generate_md5(data.get("task_id", "") + prompt)

        # 创建 metadata
        metadata = create_metadata(data)
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        # TSV 字段
        tsv_fields = [
            md5_id,
            dataset_label,
            metadata_json,
            source_b64,
            target_b64,
            prompt,  # short
            prompt,  # medium
            prompt,  # long
        ]

        return {"success": True, "tsv_fields": tsv_fields}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    input_jsonl = "/root/Z-Image/output/edit_data_from_log/edit_train_100k_filtered.jsonl"
    output_dir = "/root/paddlejob/workspace/env/vfs_benchmark_cnn/zhuyinghao/online_data_edit"
    dataset_label = "online_data_edit"
    num_workers = 8  # 降低并发数，减少SSL冲突
    lines_per_file = 10000

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 进度文件
    progress_file = output_dir / "progress.txt"
    error_file = output_dir / "errors.txt"

    # 读取所有行
    print(f"读取输入文件: {input_jsonl}")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"总行数: {total_lines}")

    # 计数器
    success_count = 0
    error_count = 0

    def write_tsv_line(idx, result):
        """写入单行 TSV"""
        nonlocal success_count, error_count

        if result["success"]:
            with write_lock:
                success_count += 1
                part_idx = success_count // lines_per_file

                part_file = output_dir / f"part-{part_idx:02d}"
                with open(part_file, 'a', encoding='utf-8') as f:
                    f.write('\t'.join(result["tsv_fields"]) + '\n')

                # 更新进度文件
                if success_count % 100 == 0:
                    with open(progress_file, 'w') as f:
                        f.write(f"{success_count}/{total_lines} ({success_count/total_lines:.1%})\n")
        else:
            with write_lock:
                error_count += 1
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"Line {idx}: {result['error']}\n")

    print("开始处理数据（边处理边写入）...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one_line, line): i for i, line in enumerate(lines)}

        for future in tqdm(as_completed(futures), total=total_lines, desc="处理中"):
            idx = futures[future]
            try:
                result = future.result()
                write_tsv_line(idx, result)
            except Exception as e:
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"Line {idx}: {e}\n")

    # 写入统计文件
    with open(output_dir / "count.txt", 'w', encoding='utf-8') as f:
        f.write(f"{dataset_label},{success_count}\n")

    # 最终进度
    with open(progress_file, 'w') as f:
        f.write(f"{success_count}/{total_lines} (完成)\n")

    print(f"\n完成！")
    print(f"成功: {success_count} 条")
    print(f"失败: {error_count} 条")
    print(f"输出目录: {output_dir}")

    # 列出生成的文件
    part_files = sorted(output_dir.glob("part-*"))
    print(f"生成 {len(part_files)} 个 part 文件")


if __name__ == "__main__":
    main()