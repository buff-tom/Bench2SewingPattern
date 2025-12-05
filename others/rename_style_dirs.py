#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="款式文件夹所在的根目录，例如 male_asia_front_and_back_garment_with_model",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="style_id_map.csv 的路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",   # ✅ 修正：只要带上 --dry-run 就为 True
        help="只打印重命名计划，不实际执行",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    csv_path = os.path.abspath(args.csv)

    print(f"Root dir : {root}")
    print(f"CSV file : {csv_path}")

    # 读取 style_id_map.csv
    mapping = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            style_id = str(row["style_id"]).strip()
            style_name = row["style_name"].strip()
            if not style_name:
                continue
            mapping.append((style_id, style_name))

    print(f"共读取 {len(mapping)} 条映射")

    # 逐个重命名
    for style_id, style_name in mapping:
        # 样式文件夹的旧名字：CSV 里的 style_name
        old_dir = os.path.join(root, style_name)
        if not os.path.isdir(old_dir):
            print(f"[WARN] 找不到目录: {old_dir}，跳过")
            continue

        # 把 style_id 补成 5 位，例如 1 -> 00001
        id_str = style_id
        if style_id.isdigit():
            id_str = f"{int(style_id):05d}"

        # 新目录名：00001_14.外套风衣中款（既有 ID 又保留中文名）
        new_name = f"{id_str}"
        new_dir = os.path.join(root, new_name)

        if os.path.exists(new_dir):
            print(f"[SKIP] 目标目录已存在: {new_dir}，跳过")
            continue

        print(f"[RENAME] {old_dir} -> {new_dir}")
        if not args.dry_run:
            os.rename(old_dir, new_dir)

    print("✅ 处理完成")


if __name__ == "__main__":
    main()
