# size_to_svg_sym.py
# -*- coding: utf-8 -*-
"""
按 Style3D 的 grade 信息生成“特定尺码”的裁片几何：
- 裁线 loops（cut_loops）
- (已简化) with_seam_loops 现在直接等于 cut_loops，不再计算缝份
- (已简化) seam_band_loops 为空

注意：已移除 pyclipper 依赖，专注于提取纯几何裁片。
"""

import os
import math
from typing import Dict, Any, List, Tuple, Optional
# from seam_offset import offset_outset, union, difference   # [移除] 不再需要复杂几何库
from collections import defaultdict

XY   = Tuple[float, float]
Loop = List[XY]
SCALE = 1000.0        # 原100 → 1000，整数化更细
ARC_TOL = 0.10        # 圆角近似精度

# -------------------- 贝塞尔曲线数学工具 --------------------
def _de_casteljau(points: List[XY], t: float) -> XY:
    """使用 De Casteljau 算法计算任意阶贝塞尔曲线在 t 时刻的坐标"""
    if len(points) == 1:
        return points[0]
    
    new_points = []
    for i in range(len(points) - 1):
        x = (1 - t) * points[i][0] + t * points[i+1][0]
        y = (1 - t) * points[i][1] + t * points[i+1][1]
        new_points.append((x, y))
        
    return _de_casteljau(new_points, t)

def _sample_bezier_curve(control_points: List[XY], segments: int = 20) -> List[XY]:
    """对贝塞尔曲线进行离散化采样"""
    if len(control_points) < 3:
        return [] 

    curve_pts = []
    for i in range(1, segments):
        t = i / float(segments)
        pt = _de_casteljau(control_points, t)
        curve_pts.append(pt)
    return curve_pts

def _poly_area(pts):
    a = 0.0
    for (x1,y1),(x2,y2) in zip(pts, pts[1:]+pts[:1]):
        a += x1*y2 - x2*y1
    return 0.5*a

# -------------------- 小工具 --------------------
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

# -------------------- grade 映射 --------------------
def build_grade_maps(by_id: Dict[int, Dict[str, Any]], grade_obj: Dict[str, Any]):
    """从 GradeGroup->grades[] 的 grade 节点读取顶点/控制点增量"""
    vertex_delta: Dict[int, XY] = {}
    ctrl_delta:   Dict[int, XY] = {}
    all_delta_by_pos: Dict[XY, XY] = {}
    
    for pair in (grade_obj.get("deltas") or []):
        if not isinstance(pair, list) or len(pair) < 2: continue
        vid, did = int(pair[0]), int(pair[1])
        g = by_id.get(did, {}) or {}
        current_pos = by_id.get(vid, {}) or {}
        v_xy = _to_xy(current_pos.get("position"))
        dv = g.get("delta") or g.get("Delta") or g.get("offset")
        xy = _to_xy(dv)
        if xy:
            vertex_delta[vid] = (float(xy[0]), float(xy[1]))
            if v_xy:
                all_delta_by_pos[(v_xy[0], v_xy[1])] = (float(xy[0]), float(xy[1]))

    for pair in (grade_obj.get("curveVertexDeltas") or []):
        if not isinstance(pair, list) or len(pair) < 2: continue
        cid, did = int(pair[0]), int(pair[1])
        g = by_id.get(did, {}) or {}
        current_pos_c = by_id.get(cid, {}) or {}
        c_xy = _to_xy(current_pos_c.get("position"))
        dv = g.get("delta") or g.get("Delta") or g.get("offset")
        xy = _to_xy(dv)
        if xy:
            ctrl_delta[cid] = (float(xy[0]), float(xy[1]))
            if c_xy:
                all_delta_by_pos[(c_xy[0], c_xy[1])] = (float(xy[0]), float(xy[1]))

    return vertex_delta, ctrl_delta, all_delta_by_pos

def get_vertex_xy(vid: int, by_id: Dict[int, Dict[str, Any]], vertex_delta: Dict[int, XY], all_delta_by_pos: Dict[XY, XY]) -> Optional[XY]:
    v = by_id.get(vid) or {}
    p = _to_xy(v.get("position") or v)
    if not p: return None
    dv = vertex_delta.get(vid, None)
    if dv is None:
        dv = all_delta_by_pos.get((p[0], p[1]), (0.0, 0.0))
    return _add(p, dv)

# -------------------- 采样边 --------------------
def sample_edge_points_grade(edge_obj: Dict[str, Any],
                             by_id: Dict[int, Dict[str, Any]],
                             vertex_delta: Dict[int, XY],
                             ctrl_delta: Dict[int, XY],
                             all_delta_by_pos: Dict[XY, XY]) -> Loop:
    """采集边的点，支持贝塞尔曲线采样"""
    if not isinstance(edge_obj, dict): return []

    pA = get_vertex_xy(int(edge_obj.get("verticeA", -1)), by_id, vertex_delta, all_delta_by_pos)
    pB = get_vertex_xy(int(edge_obj.get("verticeB", -1)), by_id, vertex_delta, all_delta_by_pos)

    curve = edge_obj.get("curve") or {}
    cps = curve.get("controlPoints")
    cps_id = edge_obj.get("curveControlPts")
    
    mid_ctrl_points = [] 

    if cps_id:
        for i in range(len(cps_id)):
            c_id = cps_id[i]
            raw_pos = None
            if isinstance(cps, list) and i+1 < len(cps): 
                 raw_pos = cps[i+1]
            elif isinstance(cps, list) and i < len(cps):
                 raw_pos = cps[i]
            
            p_xy = _to_xy(raw_pos)
            if not p_xy: continue

            dv = (0.0, 0.0)
            if c_id in ctrl_delta: dv = ctrl_delta[c_id]
            elif c_id in vertex_delta: dv = vertex_delta[c_id]
            else: dv = all_delta_by_pos.get((p_xy[0], p_xy[1]), (0.0, 0.0))
            
            mid_ctrl_points.append(_add(p_xy, dv))

    final_pts = []
    if pA: final_pts.append(pA)
    
    if not mid_ctrl_points:
        pass
    else:
        if pA and pB:
            all_bezier_pts = [pA] + mid_ctrl_points + [pB]
            curve_sampled = _sample_bezier_curve(all_bezier_pts, segments=20)
            final_pts.extend(curve_sampled)
        else:
            final_pts.extend(mid_ctrl_points)

    if pB: final_pts.append(pB)
    return final_pts

def _valid_loop(L: Loop) -> bool:
    return len(L) >= 4 and abs(_poly_area(L)) > 1e-9

def _clean_loops(loops: List[Loop]) -> List[Loop]:
    out = []
    for L in loops:
        if not L: continue
        if L[0] != L[-1]: L = L + [L[0]]
        if _valid_loop(L): out.append(L)
    return out

# -------------------- 连续边 → 裁线环 --------------------
def assemble_seqedge_loop_grade(seqedge_obj: Dict[str, Any],
                                by_id: Dict[int, Dict[str, Any]],
                                vertex_delta: Dict[int, XY],
                                ctrl_delta: Dict[int, XY],
                                all_delta_by_pos: Dict[XY, XY]) -> Loop:
    if not isinstance(seqedge_obj, dict): return []
    eids = list(seqedge_obj.get("edges") or [])
    if not eids: return []
    
    vertex_list = []
    def edge_poly(eid: int):
        eobj = by_id.get(int(eid), {}) or {}
        a = int(eobj.get("verticeA", -1)); b = int(eobj.get("verticeB", -1))
        poly = sample_edge_points_grade(eobj, by_id, vertex_delta, ctrl_delta, all_delta_by_pos)
        return a, b, poly

    a0, b0, p0 = edge_poly(eids[0])
    vertex_list.append((a0, b0))
    if len(p0) < 2: return []
    loop_pts: Loop = p0[:]

    for eid in eids[1:]:
        a, b, poly = edge_poly(eid)
        vertex_list.append((a, b))
        if len(poly) < 2: continue
        loop_pts.extend(poly)
    return loop_pts, vertex_list, eids

def pattern_to_loops_grade(pattern_obj: Dict[str, Any],
                           by_id: Dict[int, Dict[str, Any]],
                           vertex_delta: Dict[int, XY],
                           ctrl_delta: Dict[int, XY],
                           all_delta_by_pos: Dict[XY, XY]) -> List[Loop]:
    """核心几何提取：将 Pattern 对象转为坐标环列表"""
    loops: List[Loop] = []
    vertex_list_all = []
    seq_edge = {}
    if not isinstance(pattern_obj, dict): return loops
    for sid in (pattern_obj.get("sequentialEdges") or []):
        so = by_id.get(int(sid))
        if int(so.get("circleType")) != 0: continue
        L, vertex_list, eids = assemble_seqedge_loop_grade(so, by_id, vertex_delta, ctrl_delta, all_delta_by_pos)
        vertex_list_all.append(vertex_list)
        seq_edge.update({sid: eids})
        if len(L) < 4: continue
        loops.append(L)
    return loops, vertex_list_all, seq_edge

def expand_with_symmetry(by_id: Dict[int, Dict[str, Any]], piece_ids: List[int]) -> List[int]:
    out: set[int] = set(int(p) for p in piece_ids)
    for pid in list(out):
        piece = by_id.get(int(pid)) or {}
        if int(piece.get("symmetryType", 0)) == 1:
            sid = int(piece.get("symmetryClothPiece", 0) or 0)
            if sid > 0: out.add(sid)
    return list(out)

# -------------------- 主入口 (已简化) --------------------

def build_loops_for_size(by_id, size_obj, piece_ids=None, seam_join: str = "round"):
    """
    提取特定尺码的裁片几何。
    修改：完全移除缝份计算 (Seam Allowance)，仅输出裁线 (Cut Line)。
    """
    vmap, cmap, all_delta_by_pos = build_grade_maps(by_id, size_obj)
    
    if piece_ids is None:
        piece_ids = [int(p[0]) for p in (size_obj.get("clothPieceInfoMap") or []) if isinstance(p, list) and p]
    piece_ids = expand_with_symmetry(by_id, piece_ids)

    out = {}
    
    for pid in piece_ids:
        piece = by_id.get(int(pid)) or {}
        patt  = by_id.get(int(piece.get("pattern", -1))) or {}

        # 1. 提取裁线
        cut_loops, vertex_list_all, seq_edge = pattern_to_loops_grade(patt, by_id, vmap, cmap, all_delta_by_pos)
        
        if not cut_loops:
            continue
        
        # 清洗数据
        cut_loops = _clean_loops(cut_loops)
        seq_edge = {k:v for k,v in seq_edge.items() if v in cut_loops}
        
        if not cut_loops:
            continue

        # 2. 构造输出 (不再计算 offset)
        # 兼容性处理：with_seam 直接指向 cut，seamline_in 为空
        out[int(pid)] = {
            "cut": cut_loops,
            "with_seam": cut_loops,  # 兼容字段，等于净样
            "seamline_in": [],       # 无缝线
            "seq_edge": seq_edge
        }

    return out

# -------------------- 渲染与排版工具 --------------------

def bbox_of_loops(loops: List[Loop]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for L in loops:
        xs += [p[0] for p in L]
        ys += [p[1] for p in L]
    if not xs: return 0,0,0,0
    return min(xs), min(ys), max(xs), max(ys)

def translate_loops(loops: List[Loop], dx: float, dy: float) -> List[Loop]:
    return [[(x+dx, y+dy) for (x,y) in L] for L in loops]

def pack_grid(loops_by_piece: Dict[int, List[Loop]], gap: float = 60.0) -> Dict[int, List[Loop]]:
    """简单网格排版，用于生成平铺 SVG"""
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

def _render_svg_any(all_loops, out_path, fill="#0A2A6B", stroke=None, fill_opacity=1.0, stroke_w: float=0.6):
    if not all_loops: return

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

    stroke_attr = "none" if stroke is None else stroke
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' viewBox='0 0 {W:.0f} {H:.0f}'>",
        f"<path d=\"{' '.join(d_parts)}\" fill='{fill}' fill-opacity='{fill_opacity}' "
        f"stroke='{stroke_attr}' stroke-width='{stroke_w}' fill-rule='evenodd'/>",
        "</svg>",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").write('\n'.join(svg))

def render_cut_and_seamline(cut_by_piece, seamline_by_piece, out_base_path):
    """
    生成裁片 SVG (Benchmark 可视化用)
    虽然叫 cut_and_seamline，但因为 seamline_by_piece 现在通常为空，
    所以主要输出 _cut.svg。
    """
    laid_cut  = pack_grid(cut_by_piece, gap=60.0)
    
    # 获取 SVG 内容
    all_cut = []
    for pid in laid_cut:
        all_cut.extend(laid_cut[pid])
    
    if not all_cut: return

    base, ext = os.path.splitext(out_base_path)
    if not base: base = out_base_path.replace(".svg", "")
    
    # 仅输出 Cut
    _render_svg_any(all_cut, base + "_cut.svg", fill="none", stroke="#0B1A2F", stroke_w=2.0)