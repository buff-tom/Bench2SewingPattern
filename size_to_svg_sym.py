# size_to_svg_sym.py
# -*- coding: utf-8 -*-
"""
按 Style3D 的 grade 信息生成“特定尺码”的裁片几何：
- 裁线 loops（cut_loops）
- 缝份含量 loops（with_seam_loops，等于对裁线外偏移，并自动 union）
- 可选：仅缝份环带（seam_band_loops = with_seam - cut）

依赖 seam_offset.py（优先使用 pyclipper），无法使用时仍能输出裁线版。
"""

from typing import Dict, Any, List, Tuple, Optional
import os, math

from seam_offset import offset_outset, union, difference   # NEW

XY   = Tuple[float, float]
Loop = List[XY]

# --- 新增: seam 帮助函数（放在 size_to_svg_sym.py 里现有 import 之后） ---
import pyclipper as pc
from collections import defaultdict

SCALE = 1000.0        # 原100 → 1000，整数化更细
ARC_TOL = 0.10        # 新增：圆角近似精度


# ===== 新增：读取顶点 corner type（0~4） =====
def vertex_corner_mode(vid: int, by_id: dict) -> int:
    v = by_id.get(int(vid), {}) or {}
    vp = by_id.get(int(v.get("vertexProperty") or 0), {}) or {}
    # 0:Default, 1:Arc, 2:Notch, 3:Bevel, 4:Fold
    return int(vp.get("seamAllowanceVertexType", 0) or 0)

def vertex_corner_dist(vid: int, by_id: dict) -> float:
    v = by_id.get(int(vid), {}) or {}
    vp = by_id.get(int(v.get("vertexProperty") or 0), {}) or {}
    return float(vp.get("seamCornerDistance", 0.0) or 0.0)


# ===== 新增：几何小工具（方向、三角楔形、多边形布尔） =====
import math

def _poly_area(pts):
    a = 0.0
    for (x1,y1),(x2,y2) in zip(pts, pts[1:]+pts[:1]):
        a += x1*y2 - x2*y1
    return 0.5*a


def _scale_path(path):
    return [(int(round(x * SCALE)), int(round(y * SCALE))) for (x, y) in path]

# --- 新增: 读取单条边的缝份宽 ---
def edge_seam_spec(edge_obj, by_id):
    """返回 (enabled: bool, width: float) """
    eprop = by_id.get(int(edge_obj.get("edgeProperty", 0)) or 0) or {}
    if not eprop.get("enableSeamAllowance", False):
        return False, 0.0
    w = float(eprop.get("seamAllowanceWidth", 0.0) or 0.0)
    return w > 1e-6, w

# --- 新增: 为“已 grade 的一条边折线”生成开路径 buffer（两侧对称），随后再整体扣掉裁剪线即可只保留外侧部分 ---
def buffer_open_polyline(polyline, width, join_style=pc.JT_ROUND, miter_limit=2.0, arc_tol=0.25):
    if len(polyline) < 2:
        return []
    path = _scale_path(polyline)
    co = pc.PyclipperOffset(miter_limit, arc_tol * SCALE)
    # 端部用 OPEN_BUTT，避免端帽外伸造成尖刺
    co.AddPath(path, join_style, pc.ET_OPENBUTT)
    # Clipper 的 delta 是整数像素（已缩放）
    out = co.Execute(width * SCALE)
    return out  # 返回的是一个或多个封闭多边形（已缩放的整数坐标）

# -------------------- 小工具（与之前相同） --------------------
def _to_xy(v: Any) -> Optional[XY]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try: return float(v[0]), float(v[1])
        except: return None
    if isinstance(v, dict):
        p = v.get("position") or v.get("pos2D") or v.get("xy") or v.get("uv") or v.get("pos")
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try: return float(p[0]), float(p[1])
            except: return None
        x = v.get("x", v.get("u", v.get("X"))); y = v.get("y", v.get("v", v.get("Y")))
        if x is not None and y is not None:
            try: return float(x), float(y)
            except: return None
    return None


def _add(a: XY, b: XY) -> XY:
    return (a[0]+b[0], a[1]+b[1])

def _dist2(a: XY, b: XY) -> float:
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def _dedup_consecutive(pts: Loop, eps2=1e-12) -> Loop:
    out: Loop = []
    for p in pts:
        if not out or _dist2(out[-1], p) > eps2:
            out.append(p)
    return out

# -------------------- grade 映射 --------------------
def build_grade_maps(by_id: Dict[int, Dict[str, Any]], grade_obj: Dict[str, Any]):
    """从 GradeGroup->grades[] 的 grade 节点读取顶点/控制点增量"""
    vertex_delta: Dict[int, XY] = {}
    ctrl_delta:   Dict[int, XY] = {}
    all_delta_by_pos: Dict[Tuple[float, float], XY] = {}
    
    for pair in (grade_obj.get("deltas") or []):
        if not isinstance(pair, list) or len(pair) < 2: 
            continue
        vid, did = int(pair[0]), int(pair[1])
        g = by_id.get(did, {}) or {}
        current_pos = by_id.get(vid, {}) or {}
        v = current_pos.get("position")
        v_xy = _to_xy(v)
        dv = g.get("delta") or g.get("Delta") or g.get("offset")
        xy = _to_xy(dv)
        if xy:
            if by_id.get(vid) is None: continue
            vertex_delta[vid] = (float(xy[0]), float(xy[1]))
            if v_xy:
                all_delta_by_pos[(v_xy[0], v_xy[1])] = (float(xy[0]), float(xy[1]))

    for pair in (grade_obj.get("curveVertexDeltas") or []):
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        cid, did = int(pair[0]), int(pair[1])
        g = by_id.get(did, {}) or {}
        current_pos_c = by_id.get(cid, {}) or {}
        c = current_pos_c.get("position")
        c_xy = _to_xy(c)
        dv = g.get("delta") or g.get("Delta") or g.get("offset")
        xy = _to_xy(dv)
        if xy:
            if by_id.get(cid) is None: continue
            ctrl_delta[cid] = (float(xy[0]), float(xy[1]))
            if c_xy:
                all_delta_by_pos[(c_xy[0], c_xy[1])] = (float(xy[0]), float(xy[1]))

    return vertex_delta, ctrl_delta, all_delta_by_pos

def get_vertex_xy(vid: int, by_id: Dict[int, Dict[str, Any]], vertex_delta: Dict[int, XY]) -> Optional[XY]:
    v = by_id.get(vid) or {}
    p = _to_xy(v.get("position") or v)
    if not p: return None
    dv = vertex_delta.get(vid, (0.0, 0.0))
    return _add(p, dv)

def get_ctrl_xy(cid: int, by_id: Dict[int, Dict[str, Any]], ctrl_delta: Dict[int, XY], vertex_delta: Dict[int, XY]) -> Optional[XY]:
    c = by_id.get(cid) or {}
    p = _to_xy(c.get("position") or c)
    if not p: return None
    dv = ctrl_delta.get(cid, None)
    if dv is None:
        # 尝试从 vertex_delta 读取
        dv = vertex_delta.get(cid, (0.0, 0.0))
    return _add(p, dv)


# -------------------- 采样边（保持和原版一致） --------------------
def sample_edge_points_grade(edge_obj: Dict[str, Any],
                             by_id: Dict[int, Dict[str, Any]],
                             vertex_delta: Dict[int, XY],
                             ctrl_delta: Dict[int, XY]) -> Loop:
    if not isinstance(edge_obj, dict):
        return []

    pA = get_vertex_xy(int(edge_obj.get("verticeA", -1)), by_id, vertex_delta)
    pB = get_vertex_xy(int(edge_obj.get("verticeB", -1)), by_id, vertex_delta)

    cc_ids = edge_obj.get("curveControlPts")
    if isinstance(cc_ids, list) and len(cc_ids) >= 1:
        pts: Loop = []
        if pA: pts.append(pA)
        for cid in cc_ids:
            cxy = get_ctrl_xy(int(cid), by_id, ctrl_delta, vertex_delta)
            if cxy: pts.append(cxy)
        if pB: pts.append(pB)
        if len(pts) >= 2:
            return _dedup_consecutive(pts)

    curve = edge_obj.get("curve") or {}
    cps = curve.get("controlPoints")
    vertex_delta_by_pos = {key:by_id.get(key).get('position') for key in vertex_delta.keys()}
    ctrl_delta_by_pos = {key:by_id.get(key).get('position') if by_id.get(key) is not None else [0.0, 0.0, 0.0] for key in ctrl_delta.keys()}
    if isinstance(cps, list) and len(cps) >= 2:
        pts = []
        for p in cps:
            p_xy = _to_xy(p)
            if p in vertex_delta_by_pos.values():
                vid = list(vertex_delta_by_pos.keys())[list(vertex_delta_by_pos.values()).index(p)]
                dv = vertex_delta.get(vid, (0.0, 0.0))
                p_xy = _add(p_xy, dv)
            elif p in ctrl_delta_by_pos.values():
                cid = list(ctrl_delta_by_pos.keys())[list(ctrl_delta_by_pos.values()).index(p)]
                dv = ctrl_delta.get(cid, (0.0, 0.0))
                p_xy = _add(p_xy, dv)
            if p_xy: pts.append(p_xy)
        if len(pts) >= 2:
            return _dedup_consecutive(pts)

    if pA and pB:
        return [pA, pB]
    return []

def _valid_loop(L: Loop) -> bool:
    return len(L) >= 4 and abs(_poly_area(L)) > 1e-9

def _clean_loops(loops: List[Loop]) -> List[Loop]:
    out = []
    for L in loops:
        if not L: 
            continue
        if L[0] != L[-1]: 
            L = L + [L[0]]
        if _valid_loop(L):
            out.append(L)
    return out


# -------------------- 连续边 → 裁线环 --------------------
def assemble_seqedge_loop_grade(seqedge_obj: Dict[str, Any],
                                by_id: Dict[int, Dict[str, Any]],
                                vertex_delta: Dict[int, XY],
                                ctrl_delta: Dict[int, XY]) -> Loop:
    """
    严格按顶点 ID 拼接：保证上一条的 b == 下一条的 a；
    若不等则反转下一条（a<->b、折线反转），禁止用几何“最近点”去猜。
    """
    if not isinstance(seqedge_obj, dict):
        return []
    eids = list(seqedge_obj.get("edges") or [])
    if not eids:
        return []
    
    vertex_list = []

    def edge_poly(eid: int):
        eobj = by_id.get(int(eid), {}) or {}
        a = int(eobj.get("verticeA", -1)); b = int(eobj.get("verticeB", -1))
        poly = sample_edge_points_grade(eobj, by_id, vertex_delta, ctrl_delta)
        return a, b, poly

    a0, b0, p0 = edge_poly(eids[0])
    vertex_list.append(a0)
    vertex_list.append(b0)
    if len(p0) < 2:
        return []
    loop_pts: Loop = p0[:]     # 含起点

    prev_b = b0
    for eid in eids[1:]:
        a, b, poly = edge_poly(eid)
        vertex_list.append(b)
        if len(poly) < 2:
            continue
        if a != prev_b:
            # 反转这条边再接
            a, b = b, a
            poly = list(reversed(poly))
        # 去掉重复的连接点
        if _dist2(loop_pts[-1], poly[0]) <= 1e-12:
            loop_pts.extend(poly[1:])
        else:
            loop_pts.extend(poly)
        prev_b = b

    # 闭合
    if len(loop_pts) >= 3:
        if _dist2(loop_pts[0], loop_pts[-1]) > 1e-12:
            loop_pts.append(loop_pts[0])
        loop_pts = _dedup_consecutive(loop_pts)
    return loop_pts,vertex_list

def pattern_to_loops_grade(pattern_obj: Dict[str, Any],
                           by_id: Dict[int, Dict[str, Any]],
                           vertex_delta: Dict[int, XY],
                           ctrl_delta: Dict[int, XY]) -> List[Loop]:
    loops: List[Loop] = []
    vertex_list_all = []
    if not isinstance(pattern_obj, dict):
        return loops
    for sid in (pattern_obj.get("sequentialEdges") or []):
        so = by_id.get(int(sid))
        L, vertex_list = assemble_seqedge_loop_grade(so, by_id, vertex_delta, ctrl_delta)
        vertex_list_all.extend(vertex_list)
        if len(L) >= 4:
            loops.append(L)
    return loops, vertex_list_all

# ---- seam meta ----
def edge_seam_width(edge_obj: dict, by_id: dict) -> float:
    """读取一条边的缝份宽；未开启则返回0"""
    if not isinstance(edge_obj, dict):
        return 0.0
    ep_id = edge_obj.get("edgeProperty")
    if not ep_id: 
        return 0.0
    ep = by_id.get(int(ep_id), {}) or {}
    if not ep.get("enableSeamAllowance", False):
        return 0.0
    w = float(ep.get("seamAllowanceWidth", 0.0) or 0.0)
    # 也可考虑 ep.get("extraWidth") 等
    return max(0.0, w)

def vertex_corner_dist(vtx_id: int, by_id: dict) -> float:
    """读取顶点的‘缝角倒角距离’；没有就0"""
    v = by_id.get(int(vtx_id), {}) or {}
    vp_id = v.get("vertexProperty")
    if not vp_id: 
        return 0.0
    vp = by_id.get(int(vp_id), {}) or {}
    # Style3D 里常见字段名
    return float(vp.get("seamCornerDistance", 0.0) or 0.0)


def piece_uniform_seam_width(piece_obj: dict, by_id: dict) -> float:
    """遍历该片的所有连续边→边→EdgeProperty，取开启缝份的宽度中位数"""
    patt_id = piece_obj.get("pattern")
    patt = by_id.get(int(patt_id), {}) or {}
    widths = []
    for sid in (patt.get("sequentialEdges") or []):
        so = by_id.get(int(sid), {}) or {}
        for eid in (so.get("edges") or []):
            eobj = by_id.get(int(eid), {}) or {}
            w = edge_seam_width(eobj, by_id)
            if w > 1e-6:
                widths.append(w)
    if not widths:
        return 0.0
    widths.sort()
    m = widths[len(widths)//2]
    return float(m)

# -------------------- 对称片补齐 --------------------
def expand_with_symmetry(by_id: Dict[int, Dict[str, Any]],
                         piece_ids: List[int]) -> List[int]:
    out: set[int] = set(int(p) for p in piece_ids)
    for pid in list(out):
        piece = by_id.get(int(pid)) or {}
        if int(piece.get("symmetryType", 0)) == 1:
            sid = int(piece.get("symmetryClothPiece", 0) or 0)
            if sid > 0:
                out.add(sid)
    return list(out)

def piece_corner_soften_radius(piece_obj: dict, by_id: dict) -> float:
    patt = by_id.get(int(piece_obj.get("pattern", -1)), {}) or {}
    rads = []
    for sid in (patt.get("sequentialEdges") or []):
        so = by_id.get(int(sid), {}) or {}
        for eid in (so.get("edges") or []):
            eobj = by_id.get(int(eid), {}) or {}
            for vid in (eobj.get("verticeA"), eobj.get("verticeB")):
                if vid is None: continue
                d = vertex_corner_dist(int(vid), by_id)
                if d > 1e-6: rads.append(d)
    if not rads:
        return 0.0
    rads.sort()
    return float(rads[len(rads)//2])

SCALE = 100.0  # Clipper 需要整数，按需要调大

# --- 顶点属性读取 ---
def _vprop(vid: int, by_id: dict) -> dict:
    v = by_id.get(int(vid), {}) or {}
    vp_id = v.get("vertexProperty")
    return by_id.get(int(vp_id), {}) if vp_id else {}

def vertex_join_type(vid: int, by_id: dict):
    t = _vprop(vid, by_id).get("seamAllowanceVertexType", 0)
    # 0: round, 1: miter, 2: square
    return {0: "round", 1: "miter", 2: "square"}.get(int(t), "round")

def vertex_soft_radius(vid: int, by_id: dict) -> float:
    return float(_vprop(vid, by_id).get("seamCornerDistance", 0.0) or 0.0)

def vertex_snap_to_outline(vid: int, by_id: dict) -> bool:
    return bool(_vprop(vid, by_id).get("snapToOutline", False))
# 选择该条边的“最保守”连接风格（两端谁更“尖”就听谁的）

def _pc_join(join_str: str):
    return {"round": pc.JT_ROUND, "miter": pc.JT_MITER, "square": pc.JT_SQUARE}[join_str]

def buffer_edge_polyline(polyline, width, join_str="round", miter_limit=2.0, arc_tol=0.25):
    """给“单条边”的已 grade 折线做开路径缓冲（两侧对称），返回整数坐标多边形们"""
    if len(polyline) < 2 or width <= 1e-6:
        return []
    path = _scale_path(polyline)
    co = pc.PyclipperOffset(miter_limit, arc_tol * SCALE)
    co.AddPath(path, _pc_join(join_str), pc.ET_OPENBUTT)  # 边是“开路径”
    return co.Execute(width * SCALE)

def ordered_loop_edges(seqedge_obj, by_id, vmap, cmap):
    """
    返回 list[ { 'eid':int, 'a':vidA, 'b':vidB, 'poly':List[XY], 'w':float } ]
    保证相邻项满足 prev['b'] == curr['a']；若不一致就反转 poly 和 (a,b)。
    """
    out = []
    eids = list(seqedge_obj.get("edges") or [])
    if not eids: 
        return out

    # 初始取第一条边，随后的边按“能接上就用，否则反转”
    
    def edge_record(eid):
        eobj = by_id.get(int(eid), {}) or {}
        a = int(eobj.get("verticeA", -1)); b = int(eobj.get("verticeB", -1))
        poly = sample_edge_points_grade(eobj, by_id, vmap, cmap)
        w = edge_seam_width(eobj, by_id)
        return {'eid': int(eid), 'a': a, 'b': b, 'poly': poly, 'w': w, 'eobj': eobj}

    rec0 = edge_record(eids[0])
    if len(rec0['poly']) < 2:
        return out
    out.append(rec0)

    for eid in eids[1:]:
        rec = edge_record(eid)
        if len(rec['poly']) < 2:
            continue
        # 与上一条对齐
        if rec['a'] != out[-1]['b']:
            # 反转
            rec['a'], rec['b'] = rec['b'], rec['a']
            rec['poly'] = list(reversed(rec['poly']))
        out.append(rec)

    # 封环：最后一条 b 应该等于第一条 a
    if out[-1]['b'] != out[0]['a']:
        # 允许轻微错位（少数数据会有同位点重复ID），不强制修正
        pass
    return out


def build_loops_for_size(by_id, size_obj, piece_ids=None, seam_join: str = "round"):
    vmap, cmap, all_delta_by_pos = build_grade_maps(by_id, size_obj)
    print ("  Grade vertex deltas:", len(vmap), "  curve ctrl deltas:", len(cmap), "  all deltas:", len(all_delta_by_pos))
    
    if piece_ids is None:
        piece_ids = [int(p[0]) for p in (size_obj.get("clothPieceInfoMap") or []) if isinstance(p, list) and p]
    piece_ids = expand_with_symmetry(by_id, piece_ids)

    out = {}
    
    for pid in piece_ids:
        piece = by_id.get(int(pid)) or {}
        patt  = by_id.get(int(piece.get("pattern", -1))) or {}

        cut_loops, vertex_list_all = pattern_to_loops_grade(patt, by_id, vmap, cmap)
        
        # print ("  Cut loops:", len(cut_loops))
        # print (cut_loops[0])
        # print (cut_loops[1])
        # print (vertex_list_all)
        if not cut_loops:
            continue
        first_cut_loops = [cut_loops[0]]
        
        # 从数据里读“主流缝份宽”（中位数），不再猜测
        w = piece_uniform_seam_width(piece, by_id)

        if w <= 1e-6:
            out[int(pid)] = {"cut": cut_loops, "with_seam": cut_loops[:], "seam_band": [], "seamline_in": []}
            continue
        # 关键改动：对“闭合环”一次性 offset（round 避免尖刺；miter_limit 小一点）
        # _render_svg_any(first_cut_loops, out_path=f"out_temp/{size_obj.get('_name', '')}_{pid}_cut.svg")
        seam_outer   = offset_outset(cut_loops, dist=+w, join="round", miter_limit=1.2)
        seamline_in  = offset_outset(cut_loops, dist=-w, join="round", miter_limit=1.2)
        # 只要外侧缝份环带（可选）
        seam_band = difference(seam_outer, cut_loops)        
        out[int(pid)] = {
            "cut": cut_loops,
            "with_seam": cut_loops[:],
            "seam_band": seam_band,
            "seamline_in": seamline_in
        }
        seam_outer   = _clean_loops(seam_outer)
        seamline_in  = _clean_loops(seamline_in)
        

        # -- 清洗裁线 --
        cut_loops = _clean_loops(cut_loops)
        if not cut_loops:
            continue



    return out


# -------------------- 简单排版 & 渲染 --------------------

def bbox_of_loops(loops: List[Loop]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for L in loops:
        xs += [p[0] for p in L]
        ys += [p[1] for p in L]
    return min(xs), min(ys), max(xs), max(ys)

def translate_loops(loops: List[Loop], dx: float, dy: float) -> List[Loop]:
    return [[(x+dx, y+dy) for (x,y) in L] for L in loops]

def pack_grid(loops_by_piece: Dict[int, List[Loop]], gap: float = 60.0) -> Dict[int, List[Loop]]:
    pids   = list(loops_by_piece.keys())
    boxes  = {pid: bbox_of_loops(loops_by_piece[pid]) for pid in pids}
    cols   = max(1, int(len(pids)**0.5))

    x_cur, y_cur, row_h, col = 0.0, 0.0, 0.0, 0
    laid: Dict[int, List[Loop]] = {}
    for pid in pids:
        xmin, ymin, xmax, ymax = boxes[pid]
        w, h = (xmax - xmin), (ymax - ymin)
        if col == cols:
            col = 0
            x_cur = 0.0
            y_cur += row_h + gap
            row_h  = 0.0
        laid[pid] = translate_loops(loops_by_piece[pid], x_cur - xmin, y_cur - ymin)
        x_cur += w + gap
        row_h = max(row_h, h)
        col  += 1
    return laid

def render_filled_cut(loops_by_piece, out_path, color="#0A2A6B"):
    if not loops_by_piece:
        return
    laid = pack_grid(loops_by_piece, gap=60.0)
    all_loops = []
    for loops in laid.values():
        all_loops.extend(loops)
    _render_svg_any(all_loops, out_path, fill=color, stroke=None, fill_opacity=1.0)

def _render_svg_any(all_loops, out_path,
                    fill="#0A2A6B", stroke=None,  # ← stroke 改为可为 None
                    fill_opacity=1.0, stroke_w: float=0.6):
    if not all_loops:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        open(out_path, "w", encoding="utf-8").write("<svg xmlns='http://www.w3.org/2000/svg'/>")
        return

    xs, ys = [], []
    for L in all_loops:
        xs += [p[0] for p in L]; ys += [p[1] for p in L]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    pad = 50.0
    W, H = (maxx - minx + 2*pad), (maxy - miny + 2*pad)

    def TR(p): return (p[0] - minx + pad, (maxy - p[1]) + pad)

    d_parts = []
    for L in all_loops:
        pts = L if L[0] == L[-1] else L + [L[0]]
        pts2 = [TR(p) for p in pts]
        d_parts.append("M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts2) + " Z")

    # 关键：stroke none + evenodd 镂空
    stroke_attr = "none" if stroke is None else stroke
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' viewBox='0 0 {W:.0f} {H:.0f}'>",
        f"<path d=\"{' '.join(d_parts)}\" fill='{fill}' fill-opacity='{fill_opacity}' "
        f"stroke='{stroke_attr}' stroke-width='{stroke_w}' fill-rule='evenodd'/>",
        "</svg>",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").write('\n'.join(svg))


def render_combined_variant(pieces_variant: Dict[int, List[Loop]], out_path: str):
    laid = pack_grid(pieces_variant, gap=60.0)
    all_loops = []
    for v in laid.values():
        all_loops.extend(v)
    _render_svg_any(all_loops, out_path)


def render_two_versions(loops_by_piece_channels, outdir, base_name, fill_cut="#0A2A6B", fill_all="#0A2A6B"):
    os.makedirs(outdir, exist_ok=True)

    # A) 仅裁片
    all_cut = []
    for ch in loops_by_piece_channels.values():
        all_cut.extend(ch["cut"])
    _render_svg_any(all_cut, os.path.join(outdir, f"{base_name}_CUT.svg"),
               fill=fill_cut, fill_opacity=1.0, stroke="#0a2a6b", stroke_w=1.2)

    # B) 裁片 + 缝份
    all_with = []
    for ch in loops_by_piece_channels.values():
        all_with.extend(ch["with_seam"])
    _render_svg_any(all_with, os.path.join(outdir, f"{base_name}_WITH_SEAM.svg"),
               fill=fill_all, fill_opacity=1.0, stroke="#0a2a6b", stroke_w=1.2)

def render_cut_and_seamline(cut_by_piece, seamline_by_piece, out_base_path):
    """
    根据 cut_by_piece / seamline_by_piece 生成 3 个 SVG：
      <base>_cut.svg      只有裁线
      <base>_seam.svg     只有缝线
      <base>_both.svg     裁线 + 缝线叠加
    out_base_path 可以带 .svg，也可以不带。
    """
    # 统一排版
    laid_cut  = pack_grid(cut_by_piece, gap=60.0)
    laid_seam = pack_grid(seamline_by_piece, gap=60.0)

    all_cut, all_seam = [], []
    for pid in laid_cut:
        all_cut.extend(laid_cut[pid])
    for pid in laid_seam:
        all_seam.extend(laid_seam[pid])

    if not all_cut and not all_seam:
        return

    # 统一一个坐标系（后面 3 张图共用）
    xs, ys = [], []
    for L in all_cut + all_seam:
        xs += [p[0] for p in L]
        ys += [p[1] for p in L]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 50.0
    W = (maxx - minx + 2*pad)
    H = (maxy - miny + 2*pad)

    def TR(p):
        return (p[0] - minx + pad, (maxy - p[1]) + pad)

    def d_of(loops):
        if not loops:
            return ""
        parts = []
        for L in loops:
            pts = L if L[0] == L[-1] else (L + [L[0]])
            pts2 = [TR(p) for p in pts]
            parts.append(
                "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts2) + " Z"
            )
        return " ".join(parts)

    d_cut  = d_of(all_cut)
    d_seam = d_of(all_seam)
    # print(len(all_cut), len(all_seam))
    # print (len(d_cut), len(d_seam))

    # 处理输出名
    base, ext = os.path.splitext(out_base_path)
    if not base:     # 传进来只有扩展名的奇怪情况
        base = out_base_path.replace(".svg", "")
    out_cut  = base + "_cut.svg"
    out_seam = base + "_seam.svg"
    out_both = base + "_both.svg"

    os.makedirs(os.path.dirname(os.path.abspath(out_cut)) or ".", exist_ok=True)

    # 1) 只有裁线
    svg_cut = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' viewBox='0 0 {W:.0f} {H:.0f}'>",
        f"<path d=\"{d_cut}\" fill='none' stroke='#0B1A2F' stroke-width='2'/>",
        "</svg>",
    ]
    open(out_cut, "w", encoding="utf-8").write("\n".join(svg_cut))
    # print (d_cut)

    # 2) 只有缝线
    svg_seam = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' viewBox='0 0 {W:.0f} {H:.0f}'>",
        f"<path d=\"{d_seam}\" fill='none' stroke='#0A2A6B' stroke-width='0.8' stroke-dasharray='6 4'/>",
        "</svg>",
    ]
    open(out_seam, "w", encoding="utf-8").write("\n".join(svg_seam))
    # print (d_seam)

    # 3) 裁线 + 缝线叠加
    svg_both = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' viewBox='0 0 {W:.0f} {H:.0f}'>",
        f"<path d=\"{d_cut}\" fill='none' stroke='#0B1A2F' stroke-width='2'/>",
        f"<path d=\"{d_seam}\" fill='none' stroke='#4F77B0' stroke-width='0.8' stroke-dasharray='6 4'/>",
        "</svg>",
    ]
    open(out_both, "w", encoding="utf-8").write("\n".join(svg_both))


# === 新增：和 pack_grid 一样的排版，但返回每个 piece 的平移量（用于给缝线复用） ===
def pack_grid_with_shifts(loops_by_piece: Dict[int, List[Loop]], gap: float = 60.0):
    pids   = list(loops_by_piece.keys())
    boxes  = {pid: bbox_of_loops(loops_by_piece[pid]) for pid in pids}
    cols   = max(1, int(len(pids)**0.5))

    x_cur, y_cur, row_h, col = 0.0, 0.0, 0.0, 0
    laid: Dict[int, List[Loop]] = {}
    shifts: Dict[int, Tuple[float,float]] = {}
    for pid in pids:
        xmin, ymin, xmax, ymax = boxes[pid]
        w, h = (xmax - xmin), (ymax - ymin)
        if col == cols:
            col = 0
            x_cur = 0.0
            y_cur += row_h + gap
            row_h  = 0.0
        dx, dy = (x_cur - xmin), (y_cur - ymin)
        laid[pid] = translate_loops(loops_by_piece[pid], dx, dy)
        shifts[pid] = (dx, dy)  # 关键：记录 cut 的平移量
        x_cur += w + gap
        row_h = max(row_h, h)
        col  += 1
    return laid, shifts

# === 新增：把 seamline_in 放到 cut 的排版上并“深蓝填充” ===
def render_innerfill_on_cut_layout(cut_by_piece, seamline_by_piece, out_path, fill="#0A2A6B"):
    # 用 cut 的排版得到每片的平移向量
    laid_cut, shifts = pack_grid_with_shifts(cut_by_piece, gap=60.0)

    # 将 seamline_in 复用同一平移（而不是重新打包），确保与 cut 完全对齐
    laid_seam_inner_all = []
    for pid, loops in seamline_by_piece.items():
        dx, dy = shifts.get(pid, (0.0, 0.0))
        laid_seam_inner_all.extend(translate_loops(loops, dx, dy))

    # 输出纯填充（不描边），用 evenodd 规则支持有内孔的片
    _render_svg_any(laid_seam_inner_all, out_path, fill=fill, stroke=None, fill_opacity=1.0)

# === 新增：配合 debug 输出名的便捷包装 ===
def render_cut_innerfill(cut_by_piece, seamline_by_piece, out_base_path, color="#0A2A6B"):
    import os
    base, _ = os.path.splitext(out_base_path)
    out_path = base + "_cut_innerfill.svg"
    render_innerfill_on_cut_layout(cut_by_piece, seamline_by_piece, out_path, fill=color)
