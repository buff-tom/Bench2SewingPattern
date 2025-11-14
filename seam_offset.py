# seam_offset.py
# -*- coding: utf-8 -*-
"""
基于 pyclipper 的 2D 多边形外偏移 & 布尔运算工具。
输入输出均为: List[List[(x, y)]], 每个环闭合与否都可，内部会处理。
"""

from typing import List, Tuple
import math

XY = Tuple[float, float]
Loop = List[XY]
Loops = List[Loop]

_SCALE = 1000.0  # 毫米级数据建议放大到整数，避免精度问题

def _close(l: Loop) -> Loop:
    return l if (len(l) >= 3 and l[0] == l[-1]) else (l + [l[0]])

def _to_clip(paths: Loops):
    import pyclipper
    out = []
    for L in paths:
        if not L: 
            continue
        C = _close(L)
        out.append([(int(round(x*_SCALE)), int(round(y*_SCALE))) for (x,y) in C])
    return out

def _from_clip(paths):
    out: Loops = []
    for P in paths:
        if not P: 
            continue
        L: Loop = [(p[0]/_SCALE, p[1]/_SCALE) for p in P]
        # clipper 已闭合，转回开放形式更方便后续 SVG
        if len(L) >= 2 and L[0] == L[-1]:
            L = L[:-1]
        out.append(L)
    return out

def area(pts: Loop) -> float:
    a = 0.0
    for (x1,y1),(x2,y2) in zip(pts, pts[1:]+[pts[0]]):
        a += x1*y2 - x2*y1
    return 0.5*a

def orient_cw(loops: Loops) -> Loops:
    # Clipper 约定闭合多边形使用特定方向更稳定。我们统一成 CW。
    out = []
    for L in loops:
        if area(L) > 0:
            out.append(list(reversed(L)))
        else:
            out.append(L)
    return out

def union(loops: Loops) -> Loops:
    try:
        import pyclipper as pc
    except Exception:
        # 无 pyclipper：退化为原样返回
        return loops
    subj = _to_clip(orient_cw(loops))
    c = pc.Pyclipper()
    c.AddPaths(subj, pc.PT_SUBJECT, True)
    sol = c.Execute(pc.CT_UNION, pc.PFT_NONZERO, pc.PFT_NONZERO)
    return _from_clip(sol)

def difference(a: Loops, b: Loops) -> Loops:
    try:
        import pyclipper as pc
    except Exception:
        # 无 pyclipper：退化，返回 a
        return a
    subj = _to_clip(orient_cw(a))
    clip = _to_clip(orient_cw(b))
    c = pc.Pyclipper()
    c.AddPaths(subj, pc.PT_SUBJECT, True)
    c.AddPaths(clip, pc.PT_CLIP, True)
    sol = c.Execute(pc.CT_DIFFERENCE, pc.PFT_NONZERO, pc.PFT_NONZERO)
    return _from_clip(sol)

def offset_outset(loops: Loops, dist: float, join: str="round", miter_limit: float=2.0, arc_tolerance: float=0.25) -> Loops:
    """
    对闭合环整体做“向外”偏移 dist（单位与坐标一致）。
    join: "round" | "miter" | "bevel"
    """
    try:
        import pyclipper as pc
    except Exception:
        # 无 pyclipper：退化——返回原 loops（上层可选择用 SVG stroke 兜底）
        return loops

    jt = {"round": pc.JT_ROUND, "miter": pc.JT_MITER, "bevel": pc.JT_SQUARE}[join]
    po = pc.PyclipperOffset(miter_limit, arc_tolerance * _SCALE)
    po.AddPaths(_to_clip(orient_cw(loops)), jt, pc.ET_CLOSEDPOLYGON)
    sol = po.Execute(dist * _SCALE)
    return _from_clip(sol)
