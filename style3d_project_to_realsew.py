#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Style3D project.json 构建“工业级标准化”的缝合拓扑 JSON（RealSew 格式）：

主要改进：
1. **去位置化 (De-positioning)**: 强制将裁片几何重心移动到 (0,0)。
2. **布纹线对齐 (Grainline Alignment)**: 根据裁片 rotation 属性进行逆旋转，
   确保所有裁片的布纹线方向严格平行于 Y 轴。
3. **高精度几何**: 保留 size_to_svg_sym 中的贝塞尔采样精度。

输出结构：
  {
    "meta": {...},
    "pieces": [
       {
         "id": ...,
         "loops": [[x,y], ...], 
         "edges": [...],
         "normalization": { "original_center": [...], "grainline_rotation": ... }
       }
    ],
    "seams": [...]
  }
"""

import os
import json
import math
import argparse
from typing import Dict, Any, List, Tuple

# ========= 依赖模块 =========
# 确保这几个文件在同一目录下或在 PYTHONPATH 中
from json2sewing import build_sewing_topology  #
from size_to_svg_sym import (                  #
    build_grade_maps,
    pattern_to_loops_grade,
    expand_with_symmetry,
)

# ========= 几何变换工具 =========

def _calc_centroid(loops: List[List[Tuple[float, float]]]) -> Tuple[float, float]:
    """计算所有点的几何中心（简单的平均值，足以用于归一化）"""
    sum_x, sum_y, count = 0.0, 0.0, 0
    for L in loops:
        for x, y in L:
            sum_x += x
            sum_y += y
            count += 1
    if count == 0:
        return (0.0, 0.0)
    return (sum_x / count, sum_y / count)

def _rotate_point(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    """绕原点旋转二维点"""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # 标准 2D 旋转矩阵
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

def _normalize_geometry(
    loops: List[List[Tuple[float, float]]], 
    piece_rotation_degrees: float
) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
    """
    对裁片几何进行标准化处理：
    1. 布纹线对齐：应用逆旋转，使布纹线竖直。
    2. 归零：计算重心并平移到原点。
    
    返回: (处理后的 loops, 变换元数据)
    """
    # Style3D/CLO 中，rotation 通常是裁片相对于布料坐标系的旋转。
    # 为了让布纹线（通常是局部 Y 轴）变回竖直，我们需要逆向旋转。
    # 注意：角度正负需根据数据实际坐标系微调，通常逆时针为正。
    # 这里假设我们要抵消掉这个 rotation。
    angle_rad = -math.radians(piece_rotation_degrees)
    
    # 1. 旋转 (Rotate)
    rotated_loops = []
    for L in loops:
        new_L = [_rotate_point(x, y, angle_rad) for x, y in L]
        rotated_loops.append(new_L)
        
    # 2. 计算新重心 (Centroid)
    cx, cy = _calc_centroid(rotated_loops)
    
    # 3. 平移 (Translate)
    final_loops = []
    for L in rotated_loops:
        # 重心归零
        final_loops.append([(x - cx, y - cy) for x, y in L])
        
    transform_info = {
        "grainline_correction_deg": float(piece_rotation_degrees),
        "translation_offset": (cx, cy)
    }
    
    return final_loops, transform_info

# ========= 通用小工具 =========

def load_json(path: str) -> Dict[str, Any]:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, "r", encoding="latin-1"))

def build_indexes(root: Dict[str, Any]):
    all_classes: Dict[int, List[Dict[str, Any]]] = {}
    by_id: Dict[int, Dict[str, Any]] = {}
    for arr in root.get("_objectsArrays", []):
        if not isinstance(arr, list):
            continue
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            cid = obj.get("_class")
            oid = obj.get("_id")
            if cid is not None:
                all_classes.setdefault(int(cid), []).append(obj)
            if oid is not None:
                by_id[int(oid)] = obj
    return all_classes, by_id

def find_grade_group(all_classes: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any] | None:
    groups = all_classes.get(4153459189, [])
    return groups[0] if groups else None

def piece_ids_from_gradegroup(grade_group: Dict[str, Any],
                              fallback_piece_ids: List[int]) -> List[int]:
    ids = [
        int(pair[0])
        for pair in (grade_group.get("clothPieceFabricBaseMatrix") or [])
        if isinstance(pair, list) and len(pair) >= 1
    ]
    return ids if ids else [int(x) for x in (fallback_piece_ids or [])]


# ========= A. 构建 Piece (含标准化逻辑) =========

def build_pieces_and_edge_lookup_for_grade(
    by_id: Dict[int, Dict[str, Any]],
    grade_obj: Dict[str, Any],
    piece_ids_this: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    
    vmap, cmap, all_delta_by_pos = build_grade_maps(by_id, grade_obj)

    pieces_json: List[Dict[str, Any]] = []
    edge_lookup: Dict[int, Dict[str, Any]] = {}

    for pid in piece_ids_this:
        piece = by_id.get(int(pid)) or {}
        patt_id = piece.get("pattern")
        if patt_id is None:
            continue
        patt = by_id.get(int(patt_id)) or {}

        # 1. 获取原始采样坐标 (Raw Geometry)
        raw_loops, vertex_list_all, seq_edge = pattern_to_loops_grade(
            patt, by_id, vmap, cmap, all_delta_by_pos
        )
        
        if not raw_loops:
            continue

        # 2. 获取裁片旋转属性 (用于布纹线对齐)
        # Style3D 中通常叫 "rotation" (degrees) 或 "grainLine"
        # 这里默认取 key "rotation"，如果没有则为 0
        rot_deg = float(piece.get("rotation", 0.0) or 0.0)

        # 3. 标准化几何 (Normalize)
        # 执行：旋转(对齐布纹) -> 平移(重心归零)
        norm_loops, transform_meta = _normalize_geometry(raw_loops, rot_deg)

        # 4. 构建 Edge 索引 (Topology)
        local_index = 0
        edges_meta: List[Dict[str, Any]] = []

        for sid in (patt.get("sequentialEdges") or []):
            sid_int = int(sid)
            eids = seq_edge.get(sid_int) or []
            for eid in eids:
                eid_int = int(eid)
                # 记录 Edge 所属的 Piece 和 Local Index
                edge_lookup[eid_int] = {
                    "piece_id": int(pid),
                    "local_index": local_index,
                    "seqedge_id": sid_int,
                }
                edges_meta.append(
                    {
                        "edge_id": eid_int,
                        "local_index": local_index,
                        "seqedge_id": sid_int,
                    }
                )
                local_index += 1
        
        # 5. 组装 Piece 对象
        # 增加 normalization 字段，保证变换可逆
        pieces_json.append(
            {
                "id": int(pid),
                "name": piece.get("_name", str(pid)),
                "total_edges": local_index,
                "loops": [
                    {
                        "loop_id": i,
                        "role": "cut" if i == 0 else "inner", # 假设第0个是外轮廓
                        "vertices": L, # 这里已经是标准化后的坐标了
                    }
                    for i, L in enumerate(norm_loops)
                ],
                "edges": edges_meta,
                "normalization": transform_meta 
            }
        )

    return pieces_json, edge_lookup


# ========= B. 构建 Seams (保持不变) =========

def build_seams_from_sewing_pairs(
    sewing_pairs: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
    edge_lookup: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    
    seams: List[Dict[str, Any]] = []

    for cp_id, pair in sewing_pairs.items():
        (edgeA_begin, edgeA_end), (edgeB_begin, edgeB_end) = pair

        infoA_begin = edge_lookup.get(int(edgeA_begin))
        infoA_end   = edge_lookup.get(int(edgeA_end))
        infoB_begin = edge_lookup.get(int(edgeB_begin))
        infoB_end   = edge_lookup.get(int(edgeB_end))

        if not (infoA_begin and infoA_end and infoB_begin and infoB_end):
            continue

        seam = {
            "connectPair": int(cp_id),
            "a": {
                "piece_id": infoA_begin["piece_id"],
                "begin_edge_id": int(edgeA_begin),
                "end_edge_id": int(edgeA_end),
                "begin_local_index": int(infoA_begin["local_index"]),
                "end_local_index": int(infoA_end["local_index"]),
            },
            "b": {
                "piece_id": infoB_begin["piece_id"],
                "begin_edge_id": int(edgeB_begin),
                "end_edge_id": int(edgeB_end),
                "begin_local_index": int(infoB_begin["local_index"]),
                "end_local_index": int(infoB_end["local_index"]),
            },
        }
        seams.append(seam)

    return seams

# ========= C. 构建 Grade Spec =========

def build_realsew_spec_for_grade(
    data: Dict[str, Any],
    all_classes: Dict[int, List[Dict[str, Any]]],
    by_id: Dict[int, Dict[str, Any]],
    garment_obj: Dict[str, Any],
    grade_obj: Dict[str, Any],
) -> Dict[str, Any]:
    
    grade_id = int(grade_obj.get("_id"))
    size_name = grade_obj.get("_name", f"G{grade_id}")

    fallback_piece_ids = garment_obj.get("clothPieces", [])
    grade_group = find_grade_group(all_classes)
    cloth_piece_ids_global = [int(x) for x in fallback_piece_ids]
    piece_ids_this = cloth_piece_ids_global

    # 包含对称片
    piece_ids_this = expand_with_symmetry(by_id, piece_ids_this)

    # A) 几何 + 索引 (标准化已在此步完成)
    pieces, edge_lookup = build_pieces_and_edge_lookup_for_grade(by_id, grade_obj, piece_ids_this)

    # B) 拓扑
    sewing_pairs = build_sewing_topology(
        garment_obj=garment_obj,
        by_id=by_id,
        seq_edge={},
        allowed_piece_ids=piece_ids_this,
    )

    # C) 缝合
    seams = build_seams_from_sewing_pairs(sewing_pairs, edge_lookup)

    style_name = (data.get("_fileName", "proj").split("~")[0] or "proj").strip()

    spec = {
        "meta": {
            "source": "style3d_realsew",
            "style_name": style_name,
            "grade_id": grade_id,
            "grade_name": size_name,
            "unit": "mm", # size_to_svg_sym SCALE=1000 通常意味着 mm
            "coordinate_system": "cartesian_normalized",
            "notes": "Pieces are centered at (0,0) and rotated to align grainline with Y-axis."
        },
        "pieces": pieces,
        "seams": seams,
    }
    return spec


# ========= D. 批量导出 =========

def export_realsew_dataset(project_json: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    data = load_json(project_json)
    all_classes, by_id = build_indexes(data)

    garments = all_classes.get(4038497362, [])
    if not garments:
        raise RuntimeError("找不到 garment 对象 (class 4038497362)")
    garment_obj = garments[0]

    grade_group = find_grade_group(all_classes)
    if not grade_group:
        raise RuntimeError("找不到 GradeGroup (class 4153459189)")

    grade_ids = list(grade_group.get("grades") or [])
    pieces_ids = garment_obj.get("clothPieces", [])
    print(f"Processing project: {project_json} with {len(grade_ids)} sizes...")

    for gid in grade_ids:
        grade_obj = by_id.get(int(gid)) or {}
        if not grade_obj:
            continue

        spec = build_realsew_spec_for_grade(data, all_classes, by_id, garment_obj, grade_obj)
        size_name = spec["meta"]["grade_name"]
        style_name = spec["meta"]["style_name"]
        
        # 清理文件名中的非法字符
        safe_size_name = "".join([c for c in size_name if c.isalnum() or c in ('-','_')])
        
        out_path = os.path.join(outdir, f"{style_name}_{safe_size_name}_realsew.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print(f"  [OK] Exported {safe_size_name} -> {out_path}")

# ========= CLI =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_json", help="Style3D decoded project.json path")
    ap.add_argument(
        "-o", "--outdir", default="out_realsew",
        help="Output directory for normalized JSONs"
    )
    args = ap.parse_args()

    export_realsew_dataset(args.project_json, args.outdir)


if __name__ == "__main__":
    main()