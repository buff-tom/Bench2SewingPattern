#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Style3D project.json 构建“去位置化”的缝合拓扑 JSON（RealSew 格式）：

- 对每个 grade(size) 输出一个 JSON：
  {
    "meta": {...},
    "pieces": [...],
    "seams": [...]
  }

- 其中：
  pieces: 每块裁片的 cut loops + 边顺序（local_index）
  seams : 每个 connectPair 映射到 (piece, begin_edge, end_edge) 的结构化信息

同时保留一个 extract_topology_graph 接口，方便只取 (piece_definitions, topology_links)
做图学习任务。
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

# ========= 依赖你现有的模块 =========
# json2sewing: 负责从 garment_obj 中解析 connectPair → (edge, edge)
from json2sewing import build_sewing_topology  # :contentReference[oaicite:2]{index=2}

# size_to_svg_sym: 负责 grade 的放码/贝塞尔采样等
from size_to_svg_sym import (                  # :contentReference[oaicite:3]{index=3}
    build_grade_maps,
    pattern_to_loops_grade,
    expand_with_symmetry,
)

# ========= 通用小工具 =========

def load_json(path: str) -> Dict[str, Any]:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, "r", encoding="latin-1"))

def build_indexes(root: Dict[str, Any]):
    """
    构建：
      all_classes: {class_id: [obj, ...]}
      by_id:       {obj_id: obj}
    """
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
    """
    GradeGroup 的 class_id 通常是 4153459189（你之前的脚本里就是这么用的）:contentReference[oaicite:4]{index=4}
    """
    groups = all_classes.get(4153459189, [])
    return groups[0] if groups else None

def piece_ids_from_gradegroup(grade_group: Dict[str, Any],
                              fallback_piece_ids: List[int]) -> List[int]:
    """
    优先使用 GradeGroup 中 clothPieceFabricBaseMatrix 里的裁片 ID；
    否则回退到 garment.clothPieces。
    """
    ids = [
        int(pair[0])
        for pair in (grade_group.get("clothPieceFabricBaseMatrix") or [])
        if isinstance(pair, list) and len(pair) >= 1
    ]
    return ids if ids else [int(x) for x in (fallback_piece_ids or [])]


# ========= A. 每个 grade 下，构建 piece 几何 + Edge 索引 =========

def build_pieces_and_edge_lookup_for_grade(
    by_id: Dict[int, Dict[str, Any]],
    grade_obj: Dict[str, Any],
    piece_ids_this: List[int],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    对于某个 grade(size)，构建：

    - pieces_json: [ {id, name, total_edges, loops, edges}, ... ]
    - edge_lookup: {edge_id: {"piece_id": pid, "local_index": k}}

    这里完全在“裁片内部坐标系”里工作，不管排料/3D pose。
    """
    # 放码增量（对所有 piece 共用本 grade 的 deltas）
    vmap, cmap, all_delta_by_pos = build_grade_maps(by_id, grade_obj)

    pieces_json: List[Dict[str, Any]] = []
    edge_lookup: Dict[int, Dict[str, Any]] = {}

    for pid in piece_ids_this:
        piece = by_id.get(int(pid)) or {}
        patt_id = piece.get("pattern")
        if patt_id is None:
            continue
        patt = by_id.get(int(patt_id)) or {}

        # 采样该裁片当前尺码下的裁线 loops & seq_edge
        loops, vertex_list_all, seq_edge = pattern_to_loops_grade(
            patt, by_id, vmap, cmap, all_delta_by_pos
        )

        # loops 用于几何；seq_edge 用于确定 Edge 的顺序
        # local_index: 这个裁片内部所有 Edge 的顺序号（0..N-1）
        local_index = 0
        edges_meta: List[Dict[str, Any]] = []

        # pattern_obj.get("sequentialEdges") 的顺序就是边环的顺序；
        # pattern_to_loops_grade 里也是按这个顺序遍历的。:contentReference[oaicite:5]{index=5}
        for sid in (patt.get("sequentialEdges") or []):
            sid_int = int(sid)
            eids = seq_edge.get(sid_int) or []
            for eid in eids:
                eid_int = int(eid)
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

        pieces_json.append(
            {
                "id": int(pid),
                "name": piece.get("_name", str(pid)),
                "total_edges": local_index,
                "loops": [
                    {
                        "loop_id": i,
                        "role": "cut",
                        "vertices": L,
                    }
                    for i, L in enumerate(loops)
                ],
                "edges": edges_meta,
            }
        )

    return pieces_json, edge_lookup


# ========= B. 把 connectPair(sewing_pairs) 映射到 piece / local_index =========

def build_seams_from_sewing_pairs(
    sewing_pairs: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
    edge_lookup: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    sewing_pairs: connectPair_id -> ((edgeA_begin, edgeA_end), (edgeB_begin, edgeB_end))
    edge_lookup: edge_id -> {"piece_id", "local_index"}

    输出：
      seams: [
        {
          "connectPair": cp_id,
          "a": { "piece_id", "begin_edge_id", "end_edge_id",
                 "begin_local_index", "end_local_index" },
          "b": { ... }
        },
        ...
      ]
    """
    seams: List[Dict[str, Any]] = []

    for cp_id, pair in sewing_pairs.items():
        (edgeA_begin, edgeA_end), (edgeB_begin, edgeB_end) = pair

        infoA_begin = edge_lookup.get(int(edgeA_begin))
        infoA_end   = edge_lookup.get(int(edgeA_end))
        infoB_begin = edge_lookup.get(int(edgeB_begin))
        infoB_end   = edge_lookup.get(int(edgeB_end))

        # 某些 connectPair 可能引用了不在当前尺码 piece_ids_this 中的边，直接跳过
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

# ========= C. 构建某个 grade(size) 的完整 RealSew JSON =========
def build_realsew_spec_for_grade(
    data: Dict[str, Any],
    all_classes: Dict[int, List[Dict[str, Any]]],
    by_id: Dict[int, Dict[str, Any]],
    garment_obj: Dict[str, Any],
    grade_obj: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对单个 grade（单个尺码），构建完整的 RealSew spec：
      { meta, pieces, seams }
    """
    grade_id = int(grade_obj.get("_id"))
    size_name = grade_obj.get("_name", f"G{grade_id}")

    # 该服装所有 clothPieces（全局）
    fallback_piece_ids = garment_obj.get("clothPieces", [])
    grade_group = find_grade_group(all_classes)
    if grade_group:
        cloth_piece_ids_global = piece_ids_from_gradegroup(grade_group, fallback_piece_ids)
    else:
        cloth_piece_ids_global = [int(x) for x in fallback_piece_ids]

    # 本 grade 自己的 clothPieceInfoMap 优先
    piece_ids_this = [int(p[0]) for p in (grade_obj.get("clothPieceInfoMap") or []) if isinstance(p, list) and p]
    if not piece_ids_this:
        piece_ids_this = cloth_piece_ids_global

    # 加上对称片（expand_with_symmetry）
    piece_ids_this = expand_with_symmetry(by_id, piece_ids_this)

    # A) 几何 + 边索引
    pieces, edge_lookup = build_pieces_and_edge_lookup_for_grade(by_id, grade_obj, piece_ids_this)

    # B) 缝合对（connectPairs → (edge, edge)）
    sewing_pairs = build_sewing_topology(
        garment_obj=garment_obj,
        by_id=by_id,
        seq_edge={},              # 当前版本未使用该参数
        allowed_piece_ids=piece_ids_this,
    )

    # C) sewing_pairs → seams（附上 piece_id + local_index）
    seams = build_seams_from_sewing_pairs(sewing_pairs, edge_lookup)

    style_name = (data.get("_fileName", "proj").split("~")[0] or "proj").strip()

    spec = {
        "meta": {
            "source": "style3d",
            "style_name": style_name,
            "grade_id": grade_id,
            "grade_name": size_name,
        },
        "pieces": pieces,
        "seams": seams,
    }
    return spec


# ========= D. 数据集批量构建：project.json → 多个 RealSew JSON =========

def export_realsew_dataset(project_json: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    data = load_json(project_json)
    all_classes, by_id = build_indexes(data)

    # 服装对象（通常 class_id 为 4038497362，你之前脚本里也是这么用的）:contentReference[oaicite:6]{index=6}
    garments = all_classes.get(4038497362, [])
    if not garments:
        raise RuntimeError("找不到 garment 对象 (class 4038497362)")
    garment_obj = garments[0]

    grade_group = find_grade_group(all_classes)
    if not grade_group:
        raise RuntimeError("找不到 GradeGroup (class 4153459189)")

    grade_ids = list(grade_group.get("grades") or [])

    for gid in grade_ids:
        grade_obj = by_id.get(int(gid)) or {}
        if not grade_obj:
            continue

        spec = build_realsew_spec_for_grade(data, all_classes, by_id, garment_obj, grade_obj)
        size_name = spec["meta"]["grade_name"]
        style_name = spec["meta"]["style_name"]

        out_path = os.path.join(outdir, f"{style_name}_{size_name}_realsew.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {out_path}")


# ========= E. 保留你原来的 extract_topology_graph 接口（Graph 任务用） =========

def extract_topology_graph(json_path: str):
    """
    与你原来的接口兼容：
      返回 (piece_definitions, topology_links)

    - piece_definitions: {piece_id: {id, name, total_edges}}
    - topology_links: [
        {
          "u": (piece_id_u, local_index_u),
          "v": (piece_id_v, local_index_v),
          "connectPair": cp_id
        }, ...
      ]

    默认取 baseGrade（或者第一个 grade）。
    """
    data = load_json(json_path)
    all_classes, by_id = build_indexes(data)

    garments = all_classes.get(4038497362, [])
    if not garments:
        raise RuntimeError("找不到 garment 对象 (class 4038497362)")
    garment_obj = garments[0]

    grade_group = find_grade_group(all_classes)
    if not grade_group:
        raise RuntimeError("找不到 GradeGroup (class 4153459189)")

    # 优先用 baseGrade；没有就取 grades[0]
    base_grade_id = grade_group.get("baseGrade")
    if base_grade_id is not None:
        grade_obj = by_id.get(int(base_grade_id)) or {}
    else:
        grade_ids = list(grade_group.get("grades") or [])
        if not grade_ids:
            raise RuntimeError("GradeGroup 中没有 grades")
        grade_obj = by_id.get(int(grade_ids[0])) or {}

    # 复用 C 部分的逻辑，构建完整 spec（但这里只用一点信息）
    spec = build_realsew_spec_for_grade(data, all_classes, by_id, garment_obj, grade_obj)

    # 1) piece_definitions：只留 id, name, total_edges
    piece_definitions = {
        int(p["id"]): {
            "id": int(p["id"]),
            "name": p.get("name", str(p["id"])),
            "total_edges": int(p.get("total_edges", 0)),
        }
        for p in spec["pieces"]
    }

    # 2) topology_links：Graph 用的简单 (u,v)
    topology_links = []
    for seam in spec["seams"]:
        cp_id = int(seam["connectPair"])
        a = seam["a"]
        b = seam["b"]
        topology_links.append(
            {
                "u": (int(a["piece_id"]), int(a["begin_local_index"])),
                "v": (int(b["piece_id"]), int(b["begin_local_index"])),
                "connectPair": cp_id,
            }
        )

    return piece_definitions, topology_links


# ========= CLI =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_json", help="Style3D decoded project.json")
    ap.add_argument(
        "-o", "--outdir", default="out_realsew",
        help="输出的 RealSew JSON 目录"
    )
    ap.add_argument(
        "--graph-only", action="store_true",
        help="只打印 (piece_definitions, topology_links)，不导出全量 JSON"
    )
    args = ap.parse_args()

    if args.graph_only:
        piece_defs, links = extract_topology_graph(args.project_json)
        print("Pieces:", json.dumps(piece_defs, indent=2, ensure_ascii=False))
        print("Topology:", json.dumps(links, indent=2, ensure_ascii=False))
    else:
        export_realsew_dataset(args.project_json, args.outdir)


if __name__ == "__main__":
    main()
