# json2sewing.py
# -*- coding: utf-8 -*-
"""
从 Style3D project.json 中提取缝合信息：
- build_sewing_topology：返回 connectPair -> (edge, edge) 形式的缝合拓扑
- build_spec_json      ：在上面的基础上构造一个可直接保存的 spec json
"""

from typing import Dict, Any, List, Tuple

EdgeId = Tuple[int, int]                 # (begin_edge_id, end_edge_id)
SewingPairMap = Dict[int, Tuple[EdgeId, EdgeId]]  # cp_id -> ((A_begin,A_end),(B_begin,B_end))


def build_sewing_topology(
    garment_obj: Dict[str, Any],
    by_id: Dict[int, Dict[str, Any]],
    seq_edge: Dict[int, Any],
    allowed_piece_ids: List[int],
) -> SewingPairMap:
    """
    从一个 garment 对象中提取 sewing pair 信息，最终只保留 (边, 边) 的配对。

    返回:
        sewing_pairs: dict
            key: connectPair 的 _id
            val: ((edgeA_begin, edgeA_end), (edgeB_begin, edgeB_end))
    """
    if not garment_obj:
        return {}

    connectpair_ids = garment_obj.get("connectPairs") or []
    allowed_set = set(int(pid) for pid in allowed_piece_ids)
    sewing_pairs: SewingPairMap = {}

    for cp_id in connectpair_ids:
        cp_obj = by_id.get(int(cp_id))
        if not cp_obj:
            continue

        # 1) 取 SeamEdgeGroupA / B
        se_group_a_id = cp_obj.get("seamEdgeGroupA")
        se_group_b_id = cp_obj.get("seamEdgeGroupB")
        if not se_group_a_id or not se_group_b_id:
            continue

        se_group_a = by_id.get(int(se_group_a_id)) or {}
        se_group_b = by_id.get(int(se_group_b_id)) or {}
        if not se_group_a or not se_group_b:
            continue

        inst_a_ids = se_group_a.get("instancedSeamEdges") or []
        inst_b_ids = se_group_b.get("instancedSeamEdges") or []
        if not inst_a_ids or not inst_b_ids:
            continue

        # 目前只取每组里的第一个实例（以后如果有多实例再拓展）
        inst_a = by_id.get(int(inst_a_ids[0])) or {}
        inst_b = by_id.get(int(inst_b_ids[0])) or {}
        if not inst_a or not inst_b:
            continue

        # 2) 过滤不在当前尺码中的片
        piece_a_id = inst_a.get("clothPiece")
        piece_b_id = inst_b.get("clothPiece")
        if piece_a_id is None or piece_b_id is None:
            continue
        if int(piece_a_id) not in allowed_set or int(piece_b_id) not in allowed_set:
            continue

        # 3) 跟到 seamEdge → seqSegment
        seam_edge_a_id = inst_a.get("seamEdge")
        seam_edge_b_id = inst_b.get("seamEdge")
        if seam_edge_a_id is None or seam_edge_b_id is None:
            continue

        seam_edge_a = by_id.get(int(seam_edge_a_id)) or {}
        seam_edge_b = by_id.get(int(seam_edge_b_id)) or {}
        if not seam_edge_a or not seam_edge_b:
            continue

        seqseg_a_id = seam_edge_a.get("seqSegment")
        seqseg_b_id = seam_edge_b.get("seqSegment")
        if seqseg_a_id is None or seqseg_b_id is None:
            continue

        seqseg_a = by_id.get(int(seqseg_a_id)) or {}
        seqseg_b = by_id.get(int(seqseg_b_id)) or {}
        if not seqseg_a or not seqseg_b:
            continue

        # 4) seqSegment → begin / end EdgePoint
        ep_a_begin_id = seqseg_a.get("begin")
        ep_a_end_id   = seqseg_a.get("end")
        ep_b_begin_id = seqseg_b.get("begin")
        ep_b_end_id   = seqseg_b.get("end")
        if ep_a_begin_id is None or ep_a_end_id is None:
            continue
        if ep_b_begin_id is None or ep_b_end_id is None:
            continue

        ep_a_begin = by_id.get(int(ep_a_begin_id)) or {}
        ep_a_end   = by_id.get(int(ep_a_end_id))   or {}
        ep_b_begin = by_id.get(int(ep_b_begin_id)) or {}
        ep_b_end   = by_id.get(int(ep_b_end_id))   or {}
        if not ep_a_begin or not ep_a_end or not ep_b_begin or not ep_b_end:
            continue

        # 5) EdgePoint.carrier = 具体的边（Edge 的 _id）
        carrier_a_begin = ep_a_begin.get("carrier")
        carrier_a_end   = ep_a_end.get("carrier")
        carrier_b_begin = ep_b_begin.get("carrier")
        carrier_b_end   = ep_b_end.get("carrier")

        # carrier 就是 edge 的 id，这里可能出现 (同一条边, 同一条边)
        if carrier_a_begin is None or carrier_a_end is None:
            continue
        if carrier_b_begin is None or carrier_b_end is None:
            continue

        edge_a: EdgeId = (int(carrier_a_begin), int(carrier_a_end))
        edge_b: EdgeId = (int(carrier_b_begin), int(carrier_b_end))

        # 不做去重，允许 edge_a == edge_b 的情况
        sewing_pairs[int(cp_id)] = (edge_a, edge_b)

    return sewing_pairs


def build_spec_json(
    garment_obj: Dict[str, Any],
    grade_obj: Dict[str, Any],
    by_id: Dict[int, Dict[str, Any]],
    seq_edge: Dict[int, Any],
    allowed_piece_ids: List[int],
) -> Dict[str, Any]:
    """
    在 build_sewing_topology 的基础上，构造一个适合直接保存为 JSON 的 spec。

    输出示例结构：
    {
        "grade_id": 123,
        "grade_name": "S",
        "sewing_pairs": [
            {
                "connectPair": 14916,
                "edgeA": {"begin": 10522, "end": 10522},
                "edgeB": {"begin": 11001, "end": 11002}
            },
            ...
        ]
    }
    """
    sewing_pairs = build_sewing_topology(
        garment_obj=garment_obj,
        by_id=by_id,
        seq_edge=seq_edge,
        allowed_piece_ids=allowed_piece_ids,
    )

    grade_id = grade_obj.get("_id")
    size_name = grade_obj.get("_name")

    spec: Dict[str, Any] = {
        "grade_id": int(grade_id) if grade_id is not None else None,
        "grade_name": size_name,
        "sewing_pairs": [],
    }

    for cp_id, pair in sewing_pairs.items():
        (edgeA_begin, edgeA_end), (edgeB_begin, edgeB_end) = pair
        spec["sewing_pairs"].append(
            {
                "connectPair": int(cp_id),
                "edgeA": {"begin": int(edgeA_begin), "end": int(edgeA_end)},
                "edgeB": {"begin": int(edgeB_begin), "end": int(edgeB_end)},
            }
        )

    return spec
