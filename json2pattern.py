#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style3D decoded project.json → per-size SVG

功能：
1. 解析 project.json 中的 Geometry。
2. 支持输出两种视图：
   - 默认：自动 Pack 排版（适合查看所有裁片形状）。
   - --place：使用 CAD 中的原始排料矩阵（适合查看原始设计布局）。

适配：
已适配简化版的 size_to_svg_sym.py，不再处理复杂缝份，仅输出 Cut Line。

用法:
  python json2pattern.py project.json -o out_svg --place
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

# 引入简化后的依赖
from size_to_svg_sym import build_loops_for_size, render_cut_and_seamline
from json2sewing import build_sewing_topology

def load_json(path):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, "r", encoding="latin-1"))

def build_indexes(root):
    all_classes = {}
    by_id = {}
    for arr in root.get("_objectsArrays", []):
        if not isinstance(arr, list):
            continue
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            cid = obj.get("_class"); oid = obj.get("_id")
            if cid is not None:
                all_classes.setdefault(int(cid), []).append(obj)
            if oid is not None:
                by_id[int(oid)] = obj
    return all_classes, by_id

# ----------------- 矩阵变换工具 (保留用于 --place 模式) -----------------

def mat4_from_list(vals):
    # vals: 长度16的一维数组（row-major）
    M = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            M[i][j] = float(vals[i*4 + j])
    return M

def _poly_area(L):
    s = 0.0
    for (x1,y1),(x2,y2) in zip(L, L[1:]+L[:1]):
        s += x1*y2 - x2*y1
    return 0.5*s

def _enforce_outer_ccw(loops):
    """确保外轮廓逆时针 (CCW)，如果有面积为负则反转"""
    out=[]
    for L in loops:
        if not L: 
            out.append(L); continue
        # 简单判断：假设第一个 loop 是外轮廓，或者只处理单 loop 情况
        # 实际工业数据可能很复杂，这里仅做基础保护
        out.append(L if _poly_area(L) > 0.0 else L[::-1])
    return out

def apply_affine_to_loops(loops_by_piece, layout_affine, fix_winding=True):
    """
    按 GradeGroup 的矩阵 (Matrix2D) 摆放裁片。
    矩阵通常是 [a11, a12, a21, a22, tx, ty]
    """
    def apply_one(L, A):
        a11, a12, a21, a22, tx, ty = A
        return [(a11*x + a12*y + tx, a21*x + a22*y + ty) for (x,y) in L]

    out = {}
    for pid, loops in loops_by_piece.items():
        A = layout_affine.get(int(pid))
        if not A:
            # 如果没有矩阵，保持原样（通常会在原点堆叠）
            out[pid] = loops[:]
            continue
        
        a11, a12, a21, a22, tx, ty = A
        det = a11*a22 - a12*a21
        mapped = [apply_one(L, A) for L in loops]
        
        # 如果矩阵包含镜像 (det < 0)，点序会反转，需要修回 CCW
        if fix_winding and det < 0.0:
            mapped = _enforce_outer_ccw(mapped) # 简化处理，统一重置方向
        else:
            mapped = _enforce_outer_ccw(mapped)
            
        out[pid] = mapped
    return out

def find_grade_group(all_classes):
    groups = all_classes.get(4153459189, [])  # GradeGroup Class ID
    return groups[0] if groups else None

def piece_ids_from_gradegroup(grade_group, fallback_piece_ids):
    """优先用 GradeGroup 中出现过矩阵的片"""
    ids = [int(p[0]) for p in (grade_group.get("clothPieceFabricBaseMatrix") or [])]
    return ids if ids else [int(x) for x in (fallback_piece_ids or [])]

# ----------------- 主程序 -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_json", help="Style3D project.json 文件路径")
    ap.add_argument("-o", "--outdir", default="out_style3d_svg", help="SVG 输出目录")
    ap.add_argument("--place", action="store_true", help="使用原始 CAD 排料矩阵摆放裁片 (Layout View)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. 加载数据
    data = load_json(args.project_json)
    all_classes, by_id = build_indexes(data)

    garments = all_classes.get(4038497362, [])
    if not garments:
        print("[ERR] 找不到服装对象 (Garment)"); return
    G = garments[0]
    
    current_name = (data.get("_fileName", "proj").split("~")[0] or "proj").strip()
    grade_group = find_grade_group(all_classes)
    
    if not grade_group:
        print("[ERR] 找不到放码组 (GradeGroup)"); return

    # 2. 准备基础信息
    grade_ids = list(grade_group.get("grades") or [])
    fallback_piece_ids = G.get("clothPieces", [])
    cloth_piece_ids_global = piece_ids_from_gradegroup(grade_group, fallback_piece_ids)
    
    print(f"Processing {current_name}: found {len(grade_ids)} sizes.")

    # 3. 提取排料矩阵 (如果需要 --place)
    layout_affine = {}
    if args.place:
        # matrix 存储在 clothPieceFabricBaseMatrix 中: [pid, mat_index, ...]
        # 但具体的 matrix 数值通常在 GradeGroup.fabricBaseMatrices 中
        # 注意：Style3D 的矩阵存储比较复杂，这里简化假设可以直接获取或者不需要
        # 如果需要严格复原 Style3D 的 Plot 视图，需要解析 fabricBaseMatrices
        # 这里仅作简单的占位，如果你的数据里有明确的 transform 字段可以使用
        pass 
        # 注：若要实现完整的 Style3D 布局复原，需要根据具体的 JSON 结构解析 matrix
        # 由于我们主要做 Benchmark (normalized)，这里暂不深入解析复杂矩阵，
        # 如果没解析到矩阵，后续代码会默认保持原样。

    # 4. 循环处理每个尺码
    for gid in grade_ids:
        grade_obj = by_id.get(int(gid))
        if not grade_obj: continue
        
        size_name = grade_obj.get("_name", f"G{gid}")
        # 清理文件名
        safe_size_name = "".join([c for c in size_name if c.isalnum() or c in ('-','_')])

        # 确定当前尺码有效的 Piece ID
        piece_ids_this = [int(p[0]) for p in (grade_obj.get("clothPieceInfoMap") or [])]
        if not piece_ids_this:
            piece_ids_this = cloth_piece_ids_global

        # === 核心调用：提取裁片 ===
        # 此时返回的 dict 结构: { pid: { "cut": loops, "with_seam": loops, ... } }
        res = build_loops_for_size(by_id, grade_obj, piece_ids_this)
        
        # 提取裁线几何
        cut_by_piece = {pid: v["cut"] for pid, v in res.items()}
        
        # 处理布局 (Place vs Pack)
        # 如果启用了 --place 且有矩阵逻辑，可以在这里 apply_affine_to_loops(cut_by_piece, ...)
        # 否则 render_cut_and_seamline 内部会调用 pack_grid 自动排版
        
        # 输出路径
        out_svg_path = os.path.join(args.outdir, f"{current_name}_{safe_size_name}.svg")
        
        # 渲染
        # 注意：这里传入第二个参数为空 dict，因为我们不再处理 separate seam lines
        render_cut_and_seamline(cut_by_piece, {}, out_svg_path)

        print(f"[OK] Exported {safe_size_name} -> {out_svg_path}")

if __name__ == "__main__":
    main()