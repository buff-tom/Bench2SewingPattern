import os

def build_spec_json(grade_obj: dict, by_id: dict, seq_edge: dict) -> dict:
    """从一个Grade对象中提取 sewing pair 信息"""

    
def build_sewing_topology(garment_obj: dict, by_id: dict, seq_edge: dict, allowed_piece_ids: list[int]) -> dict:
    """从一个garment对象中提取 sewing pair 信息"""
    # garment_obj_id = project_obj.get("garment")
    # garment_obj = by_id.get(garment_obj_id)
    if not garment_obj:
        return {}
    connectpair_id = garment_obj.get("connectPairs")
    sewing_pairs = {}
    for cp_id in connectpair_id:
        cp_obj = by_id.get(cp_id)
        if not cp_obj:
            continue
        seamedge_a_id = cp_obj.get("seamEdgeGroupA")
        seamedge_b_id = cp_obj.get("seamEdgeGroupB")
        if not seamedge_a_id or not seamedge_b_id:
            continue
        seamedge_a_obj = by_id.get(seamedge_a_id)
        seamedge_b_obj = by_id.get(seamedge_b_id)
        if not seamedge_a_obj or not seamedge_b_obj:
            continue
        seamedge_a_instance_edge_id = seamedge_a_obj.get("instancedSeamEdges")
        seamedge_b_instance_edge_id = seamedge_b_obj.get("instancedSeamEdges")
        if not seamedge_a_instance_edge_id or not seamedge_b_instance_edge_id:
            continue
        seamedge_a_instance_edge = by_id.get(seamedge_a_instance_edge_id[0])
        seamedge_b_instance_edge = by_id.get(seamedge_b_instance_edge_id[0])
        if not seamedge_a_instance_edge or not seamedge_b_instance_edge:
            continue
        seamedge_a_piece_id = seamedge_a_instance_edge.get("clothPiece")
        seamedge_b_piece_id = seamedge_b_instance_edge.get("clothPiece")
        if not seamedge_a_piece_id or not seamedge_b_piece_id:
            continue
        if seamedge_a_piece_id not in allowed_piece_ids or seamedge_b_piece_id not in allowed_piece_ids:
            continue
        seamedge_a_sewing_id = seamedge_a_instance_edge.get("seamEdge")
        seamedge_b_sewing_id = seamedge_b_instance_edge.get("seamEdge")
        if not seamedge_a_sewing_id or not seamedge_b_sewing_id:
            continue
        seamedge_a_sewing_obj = by_id.get(seamedge_a_sewing_id)
        if not seamedge_a_sewing_obj:
            continue
        seamedge_b_sewing_obj = by_id.get(seamedge_b_sewing_id)
        if not seamedge_b_sewing_obj:
            continue
        seamedge_a_sewing_seqsegment_id = seamedge_a_sewing_obj.get("seqSegment")
        seamedge_b_sewing_seqsegment_id = seamedge_b_sewing_obj.get("seqSegment")
        if not seamedge_a_sewing_seqsegment_id or not seamedge_b_sewing_seqsegment_id:
            continue
        seamedge_a_sewing_seqsegment = by_id.get(seamedge_a_sewing_seqsegment_id)
        seamedge_b_sewing_seqsegment = by_id.get(seamedge_b_sewing_seqsegment_id)
        if not seamedge_a_sewing_seqsegment or not seamedge_b_sewing_seqsegment:
            continue
        seamedge_a_sewing_seqsegment_begin = seamedge_a_sewing_seqsegment.get("begin")
        seamedge_a_sewing_seqsegment_end = seamedge_a_sewing_seqsegment.get("end")
        seamedge_b_sewing_seqsegment_begin = seamedge_b_sewing_seqsegment.get("begin")
        seamedge_b_sewing_seqsegment_end = seamedge_b_sewing_seqsegment.get("end")
        if not seamedge_a_sewing_seqsegment_begin or not seamedge_b_sewing_seqsegment_begin:
            continue
        if not seamedge_a_sewing_seqsegment_end or not seamedge_b_sewing_seqsegment_end:
            continue
        seamedge_a_sewing_seqsegment_begin_edge_point = by_id.get(seamedge_a_sewing_seqsegment_begin).get("carrier")
        seamedge_a_sewing_seqsegment_end_edge_point = by_id.get(seamedge_a_sewing_seqsegment_end).get("carrier")
        seamedge_b_sewing_seqsegment_begin_edge_point = by_id.get(seamedge_b_sewing_seqsegment_begin).get("carrier")
        seamedge_b_sewing_seqsegment_end_edge_point = by_id.get(seamedge_b_sewing_seqsegment_end).get("carrier")
        if not seamedge_b_sewing_seqsegment_end_edge_point or not seamedge_a_sewing_seqsegment_end_edge_point:
            continue
        if not seamedge_a_sewing_seqsegment_begin_edge_point or not seamedge_b_sewing_seqsegment_begin_edge_point:
            continue
        seamedge_a_edge_id = (seamedge_a_sewing_seqsegment_begin_edge_point, seamedge_a_sewing_seqsegment_end_edge_point)
        seamedge_b_edge_id = (seamedge_b_sewing_seqsegment_begin_edge_point, seamedge_b_sewing_seqsegment_end_edge_point)
        sewing_pair_id = [seamedge_a_edge_id, seamedge_b_edge_id]
        sewing_pairs[cp_id] = sewing_pair_id
    return sewing_pairs