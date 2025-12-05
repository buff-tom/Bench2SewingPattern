#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据文件/文件夹名中的尺寸信息进行重命名，去掉多余的汉字，只保留尺寸。

示例：
  亚码L      -> L
  亚码M      -> M
  亚码S      -> S
  亚码XXL    -> XXL

用法：
  python rename_by_size.py --root /path/to/target/folder --dry-run
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

    # 正则：去除“亚码”、“L码”类似的部分，只保留 S、M、L、XL、XXL 等
    size_pattern = re.compile(r"(XXS|XS|S|M|L|XL|XXL|XXXL)")
    count = 0
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue  # 只处理目录
        count += 1
        print(f"处理目录 {count}: {name}")
        for dir_name in os.listdir(os.path.join(root, name)):
            old_path = os.path.join(root, name, dir_name)

            # 尝试匹配尺寸
            match = size_pattern.search(dir_name)
            if not match:
                continue  # 如果没有找到尺寸，就跳过这个目录

            size = match.group(1)  # 提取尺寸部分
            new_name = size  # 只保留尺寸名称

            new_path = os.path.join(root, name, new_name)

            # 如果目标名称和原名称相同，则跳过
            if new_path == old_path:
                continue

            # 如果目标目录已存在，跳过
            if os.path.exists(new_path):
                print(f"[SKIP] 目标已存在，跳过: {new_name}")
                continue

            # 打印并重命名
            print(f"[RENAME] {name} -> {new_name}")
            if not args.dry_run:
                os.rename(old_path, new_path)

    print(f"✅ 处理完成，共处理 {count} 个目录")

if __name__ == "__main__":
    main()
