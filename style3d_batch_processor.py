#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style3D Benchmark 数据集构建工具 (Batch Processor)

功能：
1. 递归遍历输入目录寻找 project.json。
2. 解析多尺码 (Grading) 数据。
3. 执行标准化：布纹线对齐 Y 轴 + 重心归零。
4. 生成 Ground Truth:
   - Spec JSON: 包含拓扑图和几何数据。
   - SVG: 可视化矢量图（自动排版）。

额外约定：
- 输出 spec / svg 直接存到「图片根目录/style_dir/size_dir」下，
  与对应的 front/back PNG 放在同一个文件夹中。
- 文件命名格式：{gender}_{region}_{styleID}_{size}_spec.json，spec 和 svg 文件放入相应的 `size_dir`。
"""

import os
import json
import math
import argparse
import traceback
import csv
import re
from typing import Dict, Any, List, Tuple
import logging

# ========= 依赖检查与导入 =========
try:
    from json2sewing import build_sewing_topology
    from size_to_svg_sym import (
        build_grade_maps,
        pattern_to_loops_grade,
        expand_with_symmetry,
        render_cut_and_seamline,  # 引用可视化渲染
        pack_grid                # 引用排版算法
    )
except ImportError as e:
    print("❌ 错误: 缺少依赖文件。请确保 json2sewing.py 和 size_to_svg_sym.py 在当前目录。")
    exit(1)

# ========= 几何标准化工具 =========

def _calc_centroid(loops: List[List[Tuple[float, float]]]) -> Tuple[float, float]:
    """计算多轮廓的几何中心"""
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
    """2D 旋转变换"""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

def _normalize_geometry(
    loops: List[List[Tuple[float, float]]], 
    piece_rotation_degrees: float
) -> Tuple[List[List[Tuple[float, float]]], Dict[str, Any]]:
    """
    标准化核心逻辑：
    1. 旋转：抵消裁片原始旋转，使布纹线垂直。
    2. 平移：将几何重心移动到原点。
    """
    # 逆向旋转以对齐布纹线到 Y 轴
    angle_rad = -math.radians(piece_rotation_degrees)
    
    # 1. 执行旋转
    rotated_loops = []
    for L in loops:
        new_L = [_rotate_point(x, y, angle_rad) for x, y in L]
        rotated_loops.append(new_L)
        
    # 2. 计算新重心
    cx, cy = _calc_centroid(rotated_loops)
    
    # 3. 执行平移归零
    final_loops = []
    for L in rotated_loops:
        final_loops.append([(x - cx, y - cy) for x, y in L])
        
    transform_info = {
        "grainline_correction_deg": float(piece_rotation_degrees),
        "translation_offset": (cx, cy)
    }
    
    return final_loops, transform_info

# ========= 数据加载辅助 =========

def load_json(path: str) -> Dict[str, Any]:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, "r", encoding="latin-1"))

def build_indexes(root: Dict[str, Any]):
    """构建 Class 和 ID 的快速索引"""
    all_classes: Dict[int, List[Dict[str, Any]]] = {}
    by_id: Dict[int, Dict[str, Any]] = {}
    for arr in root.get("_objectsArrays", []):
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict):
                    cid = obj.get("_class")
                    oid = obj.get("_id")
                    if cid is not None: all_classes.setdefault(int(cid), []).append(obj)
                    if oid is not None: by_id[int(oid)] = obj
    return all_classes, by_id

def find_grade_group(all_classes: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any] | None:
    groups = all_classes.get(4153459189, []) # Style3D GradeGroup Class ID
    return groups[0] if groups else None

def piece_ids_from_gradegroup(grade_group: Dict[str, Any], fallback_ids: List[int]) -> List[int]:
    """获取当前款式包含的所有裁片ID"""
    ids = [int(p[0]) for p in (grade_group.get("clothPieceFabricBaseMatrix") or [])]
    return ids if ids else [int(x) for x in (fallback_ids or [])]

# ========= Spec 构建逻辑 =========

def build_pieces_and_edge_lookup(
    by_id: Dict[int, Dict[str, Any]],
    grade_obj: Dict[str, Any],
    piece_ids_this: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    
    # 获取当前尺码的几何增量
    vmap, cmap, all_delta = build_grade_maps(by_id, grade_obj)

    pieces_json = []
    edge_lookup = {}

    for pid in piece_ids_this:
        piece = by_id.get(int(pid)) or {}
        patt_id = piece.get("pattern")
        if not patt_id: continue
        patt = by_id.get(int(patt_id)) or {}

        # 1. 提取原始几何 (Raw Geometry)
        raw_loops, _, seq_edge = pattern_to_loops_grade(patt, by_id, vmap, cmap, all_delta)
        if not raw_loops: continue 
        if len(raw_loops) > 1:
            raw_loops = raw_loops[:-1]
        # 2. 标准化 (Normalization)
        rot_deg = float(piece.get("rotation", 0.0) or 0.0)
        norm_loops, transform_meta = _normalize_geometry(raw_loops, rot_deg)

        # 3. 建立拓扑索引 (Edge Indexing)
        local_index = 0
        edges_meta = []
        for sid in (patt.get("sequentialEdges") or []):
            sid_int = int(sid)
            eids = seq_edge.get(sid_int) or []
            for eid in eids:
                eid_int = int(eid)
                edge_info = {
                    "piece_id": int(pid),
                    "local_index": local_index, # 裁片内的第几条边
                    "seqedge_id": sid_int,
                }
                edge_lookup[eid_int] = edge_info
                edges_meta.append({"edge_id": eid_int, **edge_info})
                local_index += 1
        
        # 4. 组装 Piece 数据
        pieces_json.append({
            "id": int(pid),
            "name": piece.get("_name", str(pid)),
            "loops": [
                {
                    "loop_id": i,
                    "type": "outer" if i == 0 else "inner",
                    "vertices": L # 已经是去位置化、布纹线对齐的坐标
                } for i, L in enumerate(norm_loops)
            ],
            "edges": edges_meta,
            "normalization": transform_meta
        })

    return pieces_json, edge_lookup

def build_seams(
    sewing_pairs: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
    edge_lookup: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """将 json2sewing 提取的原始对，转换为基于 piece_id + local_index 的图结构"""
    seams = []
    for cp_id, pair in sewing_pairs.items():
        (eA_beg, eA_end), (eB_beg, eB_end) = pair
        infoA = edge_lookup.get(int(eA_beg))
        infoB = edge_lookup.get(int(eB_beg))
        
        if infoA and infoB:
            seams.append({
                "id": int(cp_id),
                "source": {"piece": infoA["piece_id"], "edge": infoA["local_index"]},
                "target": {"piece": infoB["piece_id"], "edge": infoB["local_index"]}
            })
    return seams

def generate_visual_ground_truth(spec: Dict[str, Any], out_path_base: str):
    """
    根据生成的 Spec JSON 绘制 SVG。
    重要：JSON 数据是归零重叠的，这里调用 size_to_svg_sym 的 pack_grid 进行排版，
    生成适合视觉模型训练的平铺图 (Pattern Layout)。
    """
    cut_loops = {}
    for p in spec["pieces"]:
        # 提取坐标环
        loops = [loop["vertices"] for loop in p["loops"]]
        cut_loops[p["id"]] = loops
    
    # 使用 size_to_svg_sym 的渲染能力
    # 这会生成 _cut.svg (纯裁片) 和 _seam.svg (缝线)
    render_cut_and_seamline(cut_loops, {}, out_path_base)

# ========= 与图片数据集的映射 =========

def load_style_id_map(csv_path: str) -> Dict[str, str]:
    """
    读取 style_id_map.csv，返回:
        {style_name: '00001', ...}
    """
    mapping: Dict[str, str] = {}
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到 style_id_map.csv: {csv_path}，将不使用 style_id。")
        return mapping

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row["style_id"]).strip()
            sname = row["style_name"].strip()
            if not sname:
                continue
            if sid.isdigit():
                sid_str = f"{int(sid):05d}"
            else:
                sid_str = sid
            mapping[sname] = sid_str
    print(f"✅ 已加载 style_id_map，共 {len(mapping)} 条")
    return mapping


def find_style_dir(image_root: str, project_root: str) -> Tuple[str, str]:
    """
    在图片根目录下找到对应款式目录名，并返回:
        (style_dir_name, style_id_str)
    规则：
      
      2) 否则尝试直接用 style_name 匹配目录
      3) 最终找不到则返回 (style_name, '00000')
    """
    base_dir = os.path.basename(project_root)
    image_dir_list = os.path.abspath(image_root)
    if os.path.isdir(os.path.join(image_dir_list, base_dir)):
        
        return base_dir
    
    


def infer_prefix_from_images(size_dir: str) -> Tuple[str, str, str, str] | None:
    """
    尝试从 size 目录中的任意一张图片文件推断：
      gender, region, style_id_str, size_code
    通过解析文件名:
      gender_region_styleID_size_view.png
    """
    if not os.path.isdir(size_dir):
        return None
    for fname in os.listdir(size_dir):
        lower = fname.lower()
        if not (lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg")):
            continue
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        if len(parts) < 4:
            continue
        gender, region, sid, size = parts[:4]
        return gender, region, sid, size
    return None

# ========= 流程控制 =========

def process_single_project(
    project_root: str,
    project_path: str,
    image_root: str,
    default_gender: str,
    default_region: str,
):
    try:
        data = load_json(project_path)
        all_classes, by_id = build_indexes(data)
        
        garment = (all_classes.get(4038497362) or [{}])[0]
        grade_group = find_grade_group(all_classes)
        if not grade_group:
            return

        # Style3D 内部的款式名
        style_name = (data.get("_fileName", "proj").split("~")[0] or "proj").strip()
        if "." in style_name:
            style_name = "_".join(style_name.split("."))

        print(f"\n====== 处理款式: {style_name}  ======")

        # 在图片目录中找到对应的款式目录 + style_id
        style_dir_name = find_style_dir(image_root, project_root)
        # print(f"对应图片款式目录: {style_dir_name}")
        # print(type(style_dir_name))
        # exit(1)
        style_dir_full = os.path.join(image_root, style_dir_name)
        if not os.path.isdir(style_dir_full):
            print(f"⚠️ 款式目录不存在：{style_dir_full}，跳过该款式。")
            return

        # 确定需要处理的 Grade IDs
        grade_ids = list(grade_group.get("grades") or [])
        
        # 确定基础 Piece IDs
        fallback_ids = garment.get("clothPieces", [])

        for gid in grade_ids:
            grade_obj = by_id.get(int(gid))
            if not grade_obj:
                continue
            
            size_pattern = re.compile(r"(XXS|XS|S|M|L|XL|XXL|XXXL)")
            size_name = grade_obj.get("_name", f"G{gid}")
            match = size_pattern.search(size_name)
            if match:
                size_name = match.group(1)  # 提取尺寸部分
            else:   
                size_name = size_name  # 保持原样
            size_dir_name = size_name
            size_dir_full = os.path.join(style_dir_full, size_dir_name)

            if not os.path.isdir(size_dir_full):
                print(f"⚠️ Size 目录不存在：{size_dir_full}，将自动创建。")
                logging.info(f"创建 Size 目录：{size_dir_full}")
                os.makedirs(size_dir_full, exist_ok=True)

            # 从图片文件中推断命名前缀 {gender}_{region}_{styleID}_{size}
            prefix_info = infer_prefix_from_images(size_dir_full)
            if prefix_info is not None:
                gender_code, region_code, sid_from_img, size_code = prefix_info
            else:
                gender_code = default_gender
                region_code = default_region
                sid_from_img = style_dir_name
                size_code = size_name.upper()

            base_prefix = f"{gender_code}_{region_code}_{sid_from_img}_{size_code}"
            print(f"  -> 尺码: {size_name} | 输出前缀: {base_prefix}")

            # 1. 确定当前尺码的裁片 (含对称展开)
            pids = fallback_ids

            # 2. 构建核心 Spec 数据
            pieces, edge_map = build_pieces_and_edge_lookup(by_id, grade_obj, pids)
            
            # 3. 构建拓扑数据
            raw_sewing = build_sewing_topology(garment, by_id, {}, pids)
            seams = build_seams(raw_sewing, edge_map)

            spec = {
                "meta": {
                    "style": style_name,
                    "style_dir": style_dir_name,
                    "style_id": sid_from_img,
                    "grade": size_name,
                    "size_code": size_code,
                    "unit": "mm",
                    "coordinate_system": "normalized_centered" # 显式标记坐标系
                },
                "pieces": pieces,
                "seams": seams
            }

            # 4. 保存 Spec JSON：放在图片的 size 目录下
            json_path = os.path.join(size_dir_full, f"{base_prefix}_spec.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)

            # 5. 保存 SVG：同目录，同前缀
            svg_base = os.path.join(size_dir_full, base_prefix)
            generate_visual_ground_truth(spec, svg_base)
            
            print(f"     ✅ Spec 写入: {json_path}")
            print(f"     ✅ SVG 前缀: {svg_base}_*.svg")

    except Exception as e:
        print(f"❌ 处理失败 {project_path}: {str(e)}")
        traceback.print_exc()


def process_root(
    input_dir: str,
    image_root: str,
    default_gender: str,
    default_region: str,
):
    count = 0
    print(f"🚀 开始扫描 PRJ 目录: {input_dir}")
    for root, _, files in os.walk(input_dir):
        if "project.json" in files:
            full_path = os.path.join(root, "project.json")
            process_single_project(
                root,
                full_path,
                image_root=image_root,
                default_gender=default_gender,
                default_region=default_region,
            )
            count += 1
            
    print(f"\n✅ 批处理结束。共处理 {count} 个项目。")

# ========= 入口 =========

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Style3D Dataset Generator (绑定图片目录)")
    ap.add_argument("input_root", help="Raw Style3D data root directory (包含若干 project.json)")
    ap.add_argument(
        "-i", "--image-root",
        required=True,
        help="图片数据集根目录（包含 style 目录 / size 目录 / PNG 图片）"
    )
    ap.add_argument(
        "--gender",
        default="m",
        help="默认 gender 编码，例如 m / f（当从图片中无法推断时使用）"
    )
    ap.add_argument(
        "--region",
        default="asia",
        help="默认 region 编码，例如 asia / eur（当从图片中无法推断时使用）"
    )
    args = ap.parse_args()

    input_root = os.path.abspath(args.input_root)
    image_root = os.path.abspath(args.image_root)

    if not os.path.exists(input_root):
        print("❌ 输入 PRJ 目录不存在")
        exit(1)
    if not os.path.exists(image_root):
        print("❌ 图片根目录不存在")
        exit(1)

    process_root(
        input_dir=input_root,
        image_root=image_root,
        default_gender=args.gender,
        default_region=args.region,
    )
