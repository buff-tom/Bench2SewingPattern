#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按「款式 + 尺码」重命名图片（第二次重命名也可以）：

- 以 root 下的一级子目录名作为“款式”（style）
- style_id 由「目录名前缀数字」直接决定，例如：
    1.短袖      -> 00001
    2男长T恤    -> 00002
    21羽绒马甲  -> 00021
- 同一个款式下各个 size 用同一个 ID，只在名字里加 _S / _M / _L 等
- 新文件名格式：

    {gender}_{region}_{styleID}_{size}_{view}.ext

  例如：
    m_asia_00002_M_front.png
    m_asia_00002_M_back.png

- 只改文件名，不改目录结构
- 生成两个映射文件：
    style_id_map.csv              # 款式目录 -> styleID
    rename_mapping_by_style.csv   # 旧路径 -> 新路径 等信息
"""

import os
import sys
import csv
import argparse
import re

TARGET_EXTS = {".svg", ".json"}

# 父目录中常见的尺码写法（旧版遗留，仍可兼容）
FOLDER_SIZE_MAP = {
    "XXXL码": "XXXL",
    "XXL码": "XXL",
    "XXS码": "XXS",
    "XL码": "XL",
    "XS码": "XS",
    "L码": "L",
    "M码": "M",
    "S码": "S",
}

# 尺码关键字（长在前，避免 XL 先匹配到 L）
SIZE_KEYWORDS = [
    ("XXXL", "XXXL"),
    ("XXL",  "XXL"),
    ("XXS",  "XXS"),
    ("XL",   "XL"),
    ("XS",   "XS"),
    ("L",    "L"),
    ("M",    "M"),
    ("S",    "S"),
]


def detect_size(dirpath: str, filename: str) -> str:
    """
    优先从父目录名识别 size（兼容“亚码M/欧码L/L码...”），
    然后再从文件名里匹配，最后返回 UNK。
    """
    parent = os.path.basename(dirpath)

    # 1) 目录名中包含 XS/XL/M/S 等
    for key, val in SIZE_KEYWORDS:
        if key in parent:
            return val

    # 2) 尝试旧版的中文写法（XXL码 等）
    for k, v in FOLDER_SIZE_MAP.items():
        if k in parent:
            return v

    # 3) 从文件名中匹配
    name = os.path.splitext(filename)[0]
    for key, val in SIZE_KEYWORDS:
        if key in name:
            return val

    return "UNK"


def detect_view(filename: str) -> str:
    """识别前/后视角：前/front/back/后；否则返回空串。"""
    name = os.path.splitext(filename)[0]
    low = name.lower()

    if "前" in name or "front" in low:
        return "front"
    if "后" in name or "back" in low:
        return "back"
    # 备用：如果早期用数字区分 1 / 2，可以保留这两个分支
    if "1" in name:
        return "front"
    if "2" in name:
        return "back"
    return ""


def collect_files_group_by_style(root: str):
    """
    收集所有需要改名的文件，并按款式分组：
      style_name -> [(dirpath, filename), ...]
    其中 style_name 是 root 下的一级目录名。

    注意：这里不会再跳过以 m_asia_ / f_eur_ 开头的文件，
    方便你“二次重命名”把旧的 style_id 换成新的。
    """
    files_by_style = {}

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        # 顶层目录 '.' 说明在 root 本身；这种情况就把文件当成单独款式名 ''
        style_name = rel_dir.split(os.sep)[0] if rel_dir != "." else ""

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TARGET_EXTS:
                continue
            if fname.startswith('description'):
                continue  # 跳过 description.json 文件
            files_by_style.setdefault(style_name, []).append((dirpath, fname))

    # 去掉没有文件的 style，以及 style_name 为空的情况（通常没有）
    files_by_style = {k: v for k, v in files_by_style.items() if k}
    return files_by_style


def parse_style_id_from_name(style_name: str):
    """
    从款式目录名解析前缀数字作为 style_id：
      '2男长T恤'    -> 2
      '14.外套风衣中款' -> 14
    若解析失败返回 None。
    """
    m = re.match(r"\s*(\d+)", style_name)
    if not m:
        return None
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".",
                        help="数据根目录（包含若干款式子目录）")
    parser.add_argument("--gender", type=str, required=True,
                        help="性别编码，如 m / f")
    parser.add_argument("--region", type=str, required=True,
                        help="地区编码，如 asia / eur")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印重命名计划，不真正改名")

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    gender = args.gender.strip()
    region = args.region.strip()

    print(f"Root   : {root}")
    print(f"Gender : {gender}")
    print(f"Region : {region}")

    # 1. 收集并按款式分组
    files_by_style = collect_files_group_by_style(root)
    style_names = sorted(files_by_style.keys())

    print(f"发现 {len(style_names)} 个款式有图片")

    if not style_names:
        print("没有找到需要处理的文件，退出。")
        return

    # 2. 根据目录名前缀数字构造 style_id_map
    style_id_map = {}
    for s in style_names:
        sid = parse_style_id_from_name(s)
        if sid is None:
            # 若某个目录名没有数字前缀，则按顺序分配一个新的 ID
            sid = max(style_id_map.values(), default=0) + 1
            print(f"[WARN] 无法从款式名解析数字前缀，使用顺序 ID {sid}: {s}")
        style_id_map[s] = sid

    # 按 style_id 升序排序，便于写出 CSV
    ordered_styles = sorted(style_id_map.items(), key=lambda x: x[1])

    # 写出款式映射表
    style_map_path = os.path.join(root, "style_id_map.csv")
    with open(style_map_path, "w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["style_id", "style_name"])
        for style_name, sid in ordered_styles:
            sw.writerow([f"{sid:05d}", style_name])

    print(f"款式 ID 映射已写入: {style_map_path}")

    # 3. 重命名各款式下的文件
    mapping_path = os.path.join(root, "rename_mapping_by_style.csv")
    mf = open(mapping_path, "w", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    mw.writerow(["orig_rel_path", "new_rel_path",
                 "gender", "region", "style_id", "style_name",
                 "size", "view"])

    for style_name, sid in ordered_styles:
        files = files_by_style.get(style_name, [])
        style_id_str = f"{sid:05d}"

        print(f"\n=== 处理款式: {style_name} -> ID {style_id_str}, 共 {len(files)} 个文件 ===")

        for dirpath, fname in files:
            size = detect_size(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            base = f"{gender}_{region}_{style_id_str}_{size}"
            if ext == ".json":
                new_name = f"{base}_spec{ext}"
            else:
                new_name = f"{base}_cut{ext}"
            old_path = os.path.join(dirpath, fname)
            new_path = os.path.join(dirpath, new_name)

        

            rel_old = os.path.relpath(old_path, root)
            rel_new = os.path.relpath(new_path, root)

            print(f"{rel_old} -> {rel_new} (size={size}, ext={ext})")

            if not args.dry_run:
                os.rename(old_path, new_path)

            # mw.writerow([rel_old, rel_new, gender, region,
            #              style_id_str, style_name, size, ext])

    mf.close()
    print(f"\n✅ 重命名完成，映射文件写入: {mapping_path}")
    print(f"   款式 ID 映射见: {style_map_path}")


if __name__ == "__main__":
    main()
