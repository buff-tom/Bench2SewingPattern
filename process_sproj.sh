#!/usr/bin/env bash
# 双击运行：把当前目录下所有 .sproj 解压到 ./source_sproj/同名子目录

set -u
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ROOT_DIR="$SCRIPT_DIR/source_sproj"
mkdir -p "$ROOT_DIR"      # 先保证父目录存在

for f in *.sproj; do
    [ -f "$f" ] || continue

    base="${f%.sproj}"                # 去掉后缀
    out_dir="$ROOT_DIR/$base"         # ./source_sproj/文件名
    echo "==== 解压：$f → $out_dir/ ===="

    mkdir -p "$out_dir"               # 级联创建 out_dir
    unzip -o "$f" -d "$out_dir" >/dev/null
done

echo "==== 完成，输出在：$ROOT_DIR ===="
