#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按「款式 + 尺码」重命名图片：

- 以 root 下的一级子目录名作为“款式”（style）
- 每个款式分配一个 ID：00001, 00002, ...
- 同一个款式下各个 size 用同一个 ID，只在名字里加 _S / _M / _L
- 新文件名格式：

    {gender}_{region}_{styleID}_{size}_{view}.ext

  例如：
    f_eur_00001_S_front.png
    f_eur_00001_M_back.png

- 只改文件名，不改目录结构
- 生成两个映射文件：
    style_id_map.csv              # 款式目录 -> styleID
    rename_mapping_by_style.csv   # 旧路径 -> 新路径 等信息
"""

import os
import sys
import csv
import argparse
import random

TARGET_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 父目录中常见的尺码写法
FOLDER_SIZE_MAP = {
    "XXXL码": "XXXL",
    "XXL码": "XXL",
    "XL码": "XL",
    "XS码": "XS",
    "L码": "L",
    "M码": "M",
    "S码": "S",
}

# 文件名中常见的尺码关键字（长在前，避免 XL 先匹配到 L）
SIZE_KEYWORDS = [
    ("XXXL", "XXXL"),
    ("XXL",  "XXL"),
    ("XL",   "XL"),
    ("XS",   "XS"),
    ("L",    "L"),
    ("M",    "M"),
    ("S",    "S"),
]


def detect_size(dirpath: str, filename: str) -> str:
    """优先从父目录名识别 size，其次从文件名里匹配。"""
    parent = os.path.basename(dirpath)

    for k, v in FOLDER_SIZE_MAP.items():
        if k in parent:
            return v

    name = os.path.splitext(filename)[0]
    for k, v in SIZE_KEYWORDS:
        if k in name:
            return v

    return "UNK"


def detect_view(filename: str) -> str:
    """识别前/后视角：前/front/back/后。否则返回空串。"""
    name = os.path.splitext(filename)[0]
    low = name.lower()

    if "前" in name or "front" in low:
        return "front"
    if "后" in name or "back" in low:
        return "back"
    if "1" in name:
        return "front"
    if "2" in name:
        return "back"
    return ""


def collect_files_group_by_style(root: str, gender: str, region: str):
    """
    收集所有需要改名的文件，并按款式分组：
      style_name -> [(dirpath, filename), ...]
    其中 style_name 是 root 下的一级目录名。
    """
    prefix = f"{gender}_{region}_"
    files_by_style = {}

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        # 顶层目录 '.' 说明在 root 本身；这种情况就把文件当成单独的一个“款式名”为 ''
        style_name = rel_dir.split(os.sep)[0] if rel_dir != "." else ""

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TARGET_EXTS:
                continue

            # 已经是新规则命名的就跳过
            if fname.startswith(prefix):
                continue

            files_by_style.setdefault(style_name, []).append((dirpath, fname))

    # 去掉没有文件的 style
    files_by_style = {k: v for k, v in files_by_style.items() if v}
    # 通常我们只关心 root 下的一级子目录，所以 style_name 为空的可以视情况处理
    return files_by_style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".",
                        help="数据根目录（包含若干款式子目录）")
    parser.add_argument("--gender", type=str, required=True,
                        help="性别编码，如 m / f")
    parser.add_argument("--region", type=str, required=True,
                        help="地区编码，如 asia / eur")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（控制款式 ID 的打乱顺序）")
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
    files_by_style = collect_files_group_by_style(root, gender, region)
    style_names = sorted(k for k in files_by_style.keys() if k)  # 只用一级子目录作为款式

    print(f"发现 {len(style_names)} 个款式有图片")

    if not style_names:
        print("没有找到需要处理的文件，退出。")
        return

    # 2. 打乱款式顺序并分配 styleID（00001, 00002, ...）
    random.seed(args.seed)
    shuffled = style_names[:]
    random.shuffle(shuffled)

    style_id_map = {style_name: idx + 1 for idx, style_name in enumerate(shuffled)}

    # 写出款式映射
    style_map_path = os.path.join(root, "style_id_map.csv")
    with open(style_map_path, "w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["style_id", "style_name"])
        for s in shuffled:
            sw.writerow([f"{style_id_map[s]:05d}", s])

    print(f"款式 ID 映射已写入: {style_map_path}")

    # 3. 重命名各款式下的文件
    mapping_path = os.path.join(root, "rename_mapping_by_style.csv")
    mf = open(mapping_path, "w", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    mw.writerow(["orig_rel_path", "new_rel_path",
                 "gender", "region", "style_id", "style_name",
                 "size", "view"])

    for style_name in shuffled:
        files = files_by_style.get(style_name, [])
        style_id_int = style_id_map[style_name]
        style_id_str = f"{style_id_int:05d}"

        print(f"\n=== 处理款式: {style_name} -> ID {style_id_str}, 共 {len(files)} 个文件 ===")

        for dirpath, fname in files:
            size = detect_size(dirpath, fname)
            view = detect_view(fname)
            if view == "":
                print(f"[WARN] 无法从文件名识别视角，留空: {fname}")
            ext = os.path.splitext(fname)[1].lower()

            # 新文件名：f_eur_00001_S_front.png（view 可能为空）
            base = f"{gender}_{region}_{style_id_str}_{size}"
            if view:
                new_name = f"{base}_{view}{ext}"
            else:
                new_name = f"{base}{ext}"

            old_path = os.path.join(dirpath, fname)
            new_path = os.path.join(dirpath, new_name)

            # 防止重名
            if os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(old_path):
                suffix = 1
                base2 = base if not view else f"{base}_{view}"
                while True:
                    candidate = f"{base2}_{suffix}{ext}"
                    candidate_path = os.path.join(dirpath, candidate)
                    if not os.path.exists(candidate_path):
                        new_name = os.path.basename(candidate_path)
                        new_path = candidate_path
                        break
                    suffix += 1

            rel_old = os.path.relpath(old_path, root)
            rel_new = os.path.relpath(new_path, root)

            print(f"{rel_old} -> {rel_new} (size={size}, view={view or 'None'})")

            if not args.dry_run:
                os.rename(old_path, new_path)

            mw.writerow([rel_old, rel_new, gender, region,
                         style_id_str, style_name, size, view])

    mf.close()
    print(f"\n✅ 重命名完成，映射文件写入: {mapping_path}")
    print(f"   款式 ID 映射见: {style_map_path}")


if __name__ == "__main__":
    main()
