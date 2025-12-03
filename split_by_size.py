#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil

# 按长度从大到小匹配，避免 XS 先匹配到 S，XXL 先匹配到 XL / L
SIZE_PATTERNS = [
    ("XXXL", "XXXL码"),
    ("xxs", "XXS码"),
    ("XXS",  "XXS码"),
    ("xxl", "XXL码"),
    ("XXL",  "XXL码"),
    ("xl",  "XL码"),
    ("XL",   "XL码"),
    ("l",   "L码"),
    ("m",   "M码"),
    ("s",   "S码"),
    ("xs",  "XS码"),
    ("XS",   "XS码"),
    ("L",    "L码"),
    ("M",    "M码"),
    ("S",    "S码"),
]

def detect_size_folder(filename: str):
    """从文件名中识别尺码，并返回目标文件夹名，例如 'L码'。"""
    name = os.path.splitext(filename)[0]
    for key, folder in SIZE_PATTERNS:
        if key in name:
            return folder
    return None


def process_one_style_dir(style_dir: str):
    """
    处理单个款式目录：
    - 如果目录下有 png 文件，则根据文件名中的尺码，把文件移动到对应的尺码子目录。
    - 如果没有 png（只有 L码/M码/... 等子目录），则跳过。
    """
    items = os.listdir(style_dir)
    pngs = [
        f for f in items
        if f.lower().endswith(".png")
        and os.path.isfile(os.path.join(style_dir, f))
    ]

    # 该款式已经是“按 size 分好”的情况（只有子文件夹，没有 png），直接略过
    if not pngs:
        return

    print(f"\n=== 处理款式目录: {style_dir} ===")

    for fname in pngs:
        size_folder = detect_size_folder(fname)
        if not size_folder:
            print(f"[WARN] 无法从文件名识别尺码，跳过: {fname}")
            continue

        dst_dir = os.path.join(style_dir, size_folder)
        os.makedirs(dst_dir, exist_ok=True)

        src_path = os.path.join(style_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        if os.path.exists(dst_path):
            print(f"[SKIP] 目标已存在，跳过: {dst_path}")
            continue

        print(f"[MOVE] {src_path} -> {dst_dir}/")
        shutil.move(src_path, dst_path)


def main(root_dir: str):
    # 遍历 root_dir 下的每个款式文件夹（1.短袖、2男长T恤、...）
    for entry in os.listdir(root_dir):
        style_path = os.path.join(root_dir, entry)
        if not os.path.isdir(style_path):
            continue
        process_one_style_dir(style_path)


if __name__ == "__main__":
    # 不传参数时默认用脚本所在目录作为根目录
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = os.path.dirname(os.path.abspath(__file__))

    main(root)
    print("\n所有目录处理完成。")
