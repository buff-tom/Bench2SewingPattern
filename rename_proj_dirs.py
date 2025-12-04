#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据文件/文件夹名开头的数字进行重命名，仅保留 5 位数字编号。

示例：
  1.短袖T          -> 00001
  2男长T恤         -> 00002
  14.外套风衣中款  -> 00014
  23户外登山服-亚码 -> 00023

用法：
  python rename_by_number_prefix.py --root /path/to/source_ui --dry-run
  确认无误后去掉 --dry-run 真正执行
"""

import os
import re
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="需要重命名的目录（包含若干子文件/子文件夹）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印重命名计划，不真正执行",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f"Root: {root}")

    if not os.path.isdir(root):
        print("❌ 指定的 root 不是目录")
        return

    # 正则：匹配开头的一串数字
    pattern = re.compile(r"^\s*(\d+)") #从字符串开头匹配数字
    # pattern = re.compile(r"\D*(\d+)") #从字符串中间匹配数字


    for name in os.listdir(root):
        old_path = os.path.join(root, name)

        m = pattern.match(name)
        if not m:
            # 没有数字前缀就跳过
            continue

        num = int(m.group(1))
        id_str = f"{num:05d}"  # 补成 5 位，例如 23 -> 00023

        new_name = id_str
        new_path = os.path.join(root, new_name)

        # 已经是想要的名字就跳过
        if new_path == old_path:
            continue

        # 目标已存在就跳过，防止覆盖
        if os.path.exists(new_path):
            print(f"[SKIP] 目标已存在，跳过: {new_name}")
            continue

        print(f"[RENAME] {name} -> {new_name}")
        if not args.dry_run:
            os.rename(old_path, new_path)

    print("✅ 处理完成")


if __name__ == "__main__":
    main()
