# size_to_svg_sym.py
# -*- coding: utf-8 -*-
"""
按 Style3D 的 grade 信息生成“特定尺码”的裁片几何：
- 裁线 loops（cut_loops）
- 缝份含量 loops（with_seam_loops，等于对裁线外偏移，并自动 union）
- 可选：仅缝份环带（seam_band_loops = with_seam - cut）

依赖 seam_offset.py（优先使用 pyclipper），无法使用时仍能输出裁线版。
"""


import os
import math
import pyclipper as pc
from typing import Dict, Any, List, Tuple, Optional
from seam_offset import offset_outset, union, difference   # NEW
from collections import defaultdict

XY   = Tuple[float, float]
Loop = List[XY]
SCALE = 1000.0        # 原100 → 1000，整数化更细
ARC_TOL = 0.10        # 新增：圆角近似精度
# -------------------- 贝塞尔曲线数学工具 --------------------
def _de_casteljau(points: List[XY], t: float) -> XY:
    """
    使用 De Casteljau 算法计算任意阶贝塞尔曲线在 t 时刻的坐标。
    递归方式，支持任意数量的控制点（高阶）。
    """
    if len(points) == 1:
        return points[0]
    
    new_points = []
    for i in range(len(points) - 1):
        x = (1 - t) * points[i][0] + t * points[i+1][0]
        y = (1 - t) * points[i][1] + t * points[i+1][1]
        new_points.append((x, y))
        
    return _de_casteljau(new_points, t)

def _sample_bezier_curve(control_points: List[XY], segments: int = 20) -> List[XY]:
    """
    输入：[起点, 控制点1, 控制点2, ..., 终点]
    输出：采样后的曲线点列表（不含起点和终点，避免拼接时重复）
    segments: 采样段数，越高越平滑
    """
    if len(control_points) < 3:
        return [] # 直线无需采样中间点

    curve_pts = []
    # 从 t > 0 到 t < 1 进行采样 (不包含 0 和 1)
    # t=0 是起点，t=1 是终点，这两点由外部逻辑添加
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

# -------------------- grade 映射 --------------------
def build_grade_maps(by_id: Dict[int, Dict[str, Any]], grade_obj: Dict[str, Any]):
    """从 GradeGroup->grades[] 的 grade 节点读取顶点/控制点增量"""
    vertex_delta: Dict[int, XY] = {}
    ctrl_delta:   Dict[int, XY] = {}
    all_delta_by_pos: Dict[XY, XY] = {}
    
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

def get_vertex_xy(vid: int, by_id: Dict[int, Dict[str, Any]], vertex_delta: Dict[int, XY], all_delta_by_pos: Dict[XY, XY]) -> Optional[XY]:
    v = by_id.get(vid) or {}
    p = _to_xy(v.get("position") or v)
    if not p: return None
    dv = vertex_delta.get(vid, None)
    if dv is None:
        # 尝试从 all_delta_by_pos 读取
        dv = all_delta_by_pos.get((p[0], p[1]), (0.0, 0.0))
    return _add(p, dv)

def get_ctrl_xy(cid: int, by_id: Dict[int, Dict[str, Any]], ctrl_delta: Dict[int, XY], vertex_delta: Dict[int, XY], all_delta_by_pos: Dict[XY, XY]) -> Optional[XY]:
    c = by_id.get(cid) or {}
    p = _to_xy(c.get("position") or c)
    if not p: return None
    dv = ctrl_delta.get(cid, None)
    if dv is None:
        # 尝试从 vertex_delta 读取
        dv = vertex_delta.get(cid, None)
        if dv is None:
            # 尝试从 all_delta_by_pos 读取
            dv = all_delta_by_pos.get((p[0], p[1]), (0.0, 0.0))
    return _add(p, dv)


# -------------------- 采样边（保持和原版一致） --------------------
# -------------------- 采样边（修改版：支持贝塞尔曲线） --------------------
# -------------------- 采样边（升级版：支持任意阶贝塞尔曲线） --------------------
def sample_edge_points_grade(edge_obj: Dict[str, Any],
                             by_id: Dict[int, Dict[str, Any]],
                             vertex_delta: Dict[int, XY],
                             ctrl_delta: Dict[int, XY],
                             all_delta_by_pos: Dict[XY, XY]) -> Loop:
    if not isinstance(edge_obj, dict):
        return []

    # 1. 获取起点 A
    pA = get_vertex_xy(int(edge_obj.get("verticeA", -1)), by_id, vertex_delta, all_delta_by_pos)
    # 2. 获取终点 B
    pB = get_vertex_xy(int(edge_obj.get("verticeB", -1)), by_id, vertex_delta, all_delta_by_pos)

    # 3. 收集所有中间控制点（应用放码增量）
    curve = edge_obj.get("curve") or {}
    cps = curve.get("controlPoints")      # 坐标列表
    cps_id = edge_obj.get("curveControlPts") # ID列表
    
    mid_ctrl_points = [] # 存储纯坐标 [(x,y), ...]

    if cps_id:
        # Style3D/CLO 数据中，controlPoints 通常包含了所有点，或者与 curveControlPts 索引对应
        # 这里我们严谨地通过 ID 查找对应的原始坐标，并加上增量
        for i in range(len(cps_id)):
            c_id = cps_id[i]
            
            # 尝试获取该控制点的基础坐标
            # 注意：cps 结构可能是 list 也可能是 dict，视解析器而定
            # 如果 cps 是 list，通常第0个是起点(或无效)，第1个开始是控制点，具体视数据版本而定
            # 最稳妥的方式是：如果有 ID，尝试去 ctrl_delta 查；如果没有，尝试从 geometry 读
            
            # 尝试从 cps 列表读取原始坐标作为 Base
            raw_pos = None
            if isinstance(cps, list) and i+1 < len(cps): 
                 # 许多格式中 cps[0] 是起点，cps[i+1] 是第i个控制点
                 raw_pos = cps[i+1]
            elif isinstance(cps, list) and i < len(cps):
                 raw_pos = cps[i]
            
            p_xy = _to_xy(raw_pos)
            
            # 如果找不到原始坐标，甚至可以尝试去 by_id 找 (虽然控制点通常不在 by_id 顶级索引中)
            if not p_xy:
                # 最后的兜底：如果无法定位基础位置，跳过该点（防止报错）
                continue

            # 计算增量 (Grade Delta)
            dv = (0.0, 0.0)
            if c_id in ctrl_delta:
                dv = ctrl_delta[c_id]
            elif c_id in vertex_delta:
                dv = vertex_delta[c_id]
            else:
                # 尝试空间位置匹配
                dv = all_delta_by_pos.get((p_xy[0], p_xy[1]), (0.0, 0.0))
            
            # 叠加增量
            final_xy = _add(p_xy, dv)
            mid_ctrl_points.append(final_xy)

    # 4. 构建最终点序列
    final_pts = []
    
    # 加入起点
    if pA: 
        final_pts.append(pA)
    
    # 处理中间部分
    if not mid_ctrl_points:
        # 直线：没有中间控制点，直接连过去
        pass
    else:
        # 曲线：必须有起点和终点才能生成
        if pA and pB:
            # 组合所有点：[起点, 控制点1, 控制点2, ..., 终点]
            all_bezier_pts = [pA] + mid_ctrl_points + [pB]
            
            # 进行采样 (segments=20 足够平滑，如需更高精度可调大)
            curve_sampled = _sample_bezier_curve(all_bezier_pts, segments=20)
            
            final_pts.extend(curve_sampled)
        else:
            # 异常情况：有控制点但缺端点，降级为直接连接控制点（为了画出来debug）
            final_pts.extend(mid_ctrl_points)

    # 加入终点
    if pB: 
        final_pts.append(pB)

    return final_pts

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
                                ctrl_delta: Dict[int, XY],
                                all_delta_by_pos: Dict[XY, XY]) -> Loop:
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
        poly = sample_edge_points_grade(eobj, by_id, vertex_delta, ctrl_delta, all_delta_by_pos)
        return a, b, poly

    a0, b0, p0 = edge_poly(eids[0])
    vertex_list.append((a0, b0))
    if len(p0) < 2:
        return []
    loop_pts: Loop = p0[:]     # 含起点

    for eid in eids[1:]:
        a, b, poly = edge_poly(eid)
        vertex_list.append((a, b))
        if len(poly) < 2:
            continue
        # 去掉重复的连接点
        loop_pts.extend(poly)
    return loop_pts, vertex_list, eids

def pattern_to_loops_grade(pattern_obj: Dict[str, Any],
                           by_id: Dict[int, Dict[str, Any]],
                           vertex_delta: Dict[int, XY],
                           ctrl_delta: Dict[int, XY],
                           all_delta_by_pos: Dict[XY, XY]) -> List[Loop]:
    loops: List[Loop] = []
    vertex_list_all = []
    seq_edge = {}
    if not isinstance(pattern_obj, dict):
        return loops
    for sid in (pattern_obj.get("sequentialEdges") or []):
        so = by_id.get(int(sid))
        if int(so.get("circleType")) != 0:
            continue
        L, vertex_list, eids = assemble_seqedge_loop_grade(so, by_id, vertex_delta, ctrl_delta, all_delta_by_pos)
        vertex_list_all.append(vertex_list)
        seq_edge.update({sid: eids})
        if len(L) < 4:
            continue
        loops.append(L)
    return loops, vertex_list_all, seq_edge

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

        cut_loops, vertex_list_all, seq_edge = pattern_to_loops_grade(patt, by_id, vmap, cmap, all_delta_by_pos)
        
        
        # print (f"{vertex_list_all}")
        # print ("  Cut loops:", len(cut_loops))
        # print (cut_loops[0])
        # print (cut_loops[1])
        # print (vertex_list_all)
        if not cut_loops:
            continue
        cut_loops = cut_loops[:-1]
        seq_edge = {k:v for k,v in seq_edge.items() if v in cut_loops}
        
        # 从数据里读“主流缝份宽”（中位数），不再猜测
        w = piece_uniform_seam_width(piece, by_id)

        if w <= 1e-6:
            out[int(pid)] = {"cut": cut_loops, "with_seam": cut_loops[:], "seam_band": [], "seamline_in": []}
            continue
        # 关键改动：对“闭合环”一次性 offset（round 避免尖刺；miter_limit 小一点）
        # _render_svg_any(first_cut_loops, out_path=f"out_temp/{size_obj.get('_name', '')}_{pid}_cut.svg")
        seam_outer   = offset_outset(cut_loops, dist=+w, join="round", miter_limit=1.2)
        seam_outer = _clean_loops(seam_outer)  # 清洗后赋值
        seamline_in  = offset_outset(cut_loops, dist=-w, join="round", miter_limit=1.2)
        seamline_in = _clean_loops(seamline_in)  # 清洗后赋值
        # 只要外侧缝份环带（可选）
        seam_band = difference(seam_outer, cut_loops)        
        out[int(pid)] = {
            "cut": cut_loops,
            "with_seam": seam_outer,
            "seam_band": seam_band,
            "seamline_in": seamline_in,
            "seq_edge": seq_edge
        }
        seamline_in  = _clean_loops(seamline_in)
        

        # -- 清洗裁线 --
        cut_loops = _clean_loops(cut_loops)
        seq_edge = {k:v for k,v in seq_edge.items() if v in cut_loops}
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