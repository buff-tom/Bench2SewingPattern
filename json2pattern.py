#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style3D decoded project.json → per-size SVG：
- cut：仅裁线填充
- with_seam：整片含缝份填充
- (可选) seam_band：只显示缝份环带

用法:
  python style3d_project_to_svg.py project.json -o out --variant both

依赖:
  pip install pyclipper
"""
import os, json, argparse
from collections import defaultdict

from size_to_svg_sym import build_loops_for_size, render_combined_variant, render_cut_and_seamline

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
            all_classes.setdefault(cid, []).append(obj)
            if oid is not None:
                by_id[oid] = obj
    return all_classes, by_id

def translate_text(
    translate_client,
    text: str | bytes | list[str] = "¡Hola amigos y amigas!",
    target_language: str = "en",
    source_language: str | None = None,
) -> dict:
    """Translates a given text into the specified target language.

    Find a list of supported languages and codes here:
    https://cloud.google.com/translate/docs/languages#nmt

    Args:
        text: The text to translate. Can be a string, bytes or a list of strings.
              If bytes, it will be decoded as UTF-8.
        target_language: The ISO 639 language code to translate the text into
                         (e.g., 'en' for English, 'es' for Spanish).
        source_language: Optional. The ISO 639 language code of the input text
                         (e.g., 'fr' for French). If None, the API will attempt
                         to detect the source language automatically.

    Returns:
        A dictionary containing the translation results.
    """

    if isinstance(text, bytes):
        text = [text.decode("utf-8")]

    if isinstance(text, str):
        text = [text]

    # If a string is supplied, a single dictionary will be returned.
    # In case a list of strings is supplied, this method
    # will return a list of dictionaries.

    # Find more information about translate function here:
    # https://cloud.google.com/python/docs/reference/translate/latest/google.cloud.translate_v2.client.Client#google_cloud_translate_v2_client_Client_translate
    results = translate_client.translate(
        values=text,
        target_language=target_language,
        source_language=source_language
    )

    # for result in results:
    #     if "detectedSourceLanguage" in result:
    #         print(f"Detected source language: {result['detectedSourceLanguage']}")

        # print(f"Input text: {result['input']}")
        # print(f"Translated text: {result['translatedText']}")
        # print()
    # print(results)
    return results

def mat4_from_list(vals):
    # vals: 长度16的一维数组（row-major）
    M = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            M[i][j] = float(vals[i*4 + j])
    return M

def apply_matrix2D_via_X0Y(M, loops):
    # 将2D点(x,y)→3D(x,0,y)→乘M→取(X,Z)回到平面
    out = []
    for L in loops:
        LL = []
        for (x,y) in L:
            X = M[0][0]*x + M[0][1]*0.0 + M[0][2]*y + M[0][3]*1.0
            Y = M[1][0]*x + M[1][1]*0.0 + M[1][2]*y + M[1][3]*1.0
            Z = M[2][0]*x + M[2][1]*0.0 + M[2][2]*y + M[2][3]*1.0
            LL.append((X, Z))   # 用 (X,Z) 作平面渲染
        out.append(LL)
    return out

def build_pieceid_to_gradeid(size_obj):
    mp = {}
    for pair in (size_obj.get("clothPieceInfoMap") or []):
        if isinstance(pair, list) and len(pair) >= 2:
            mp[int(pair[0])] = int(pair[1])
    return mp

# --- 解析 GradeGroup & 取2D仿射 ---
def find_grade_group(all_classes):
    groups = all_classes.get(4153459189, [])  # GradeGroup
    return groups[0] if groups else None

def mat4_to_affine2d(m16):
    # 4x4（行主序） → 2D 仿射 (a11,a12,a21,a22,tx,ty)
    a11,a12 = float(m16[0]),  float(m16[1])
    a21,a22 = float(m16[4]),  float(m16[5])
    tx, ty  = float(m16[12]), float(m16[13])
    return (a11,a12,a21,a22,tx,ty)

def _poly_area(L):
    s=0.0
    for (x1,y1),(x2,y2) in zip(L, L[1:]+L[:1]):
        s += x1*y2 - x2*y1
    return 0.5*s

def _enforce_outer_ccw(loops):
    # 单层外轮廓为主的常见纸样：面积<0 则反转点序
    out=[]
    for L in loops:
        if not L: 
            out.append(L); continue
        out.append(L if _poly_area(L) > 0.0 else L[::-1])
    return out

def apply_affine_to_loops(loops_by_piece, layout_affine, fix_winding=True):
    """按 GradeGroup 的矩阵摆放；若 det<0（镜像），统一反转点序保持外环 CCW"""
    def apply_one(L, A):
        a11,a12,a21,a22,tx,ty = A
        return [(a11*x + a12*y + tx, a21*x + a22*y + ty) for (x,y) in L]

    out = {}
    for pid, loops in loops_by_piece.items():
        A = layout_affine.get(int(pid))
        if not A:
            out[pid] = loops[:]
            continue
        a11,a12,a21,a22,tx,ty = A
        det = a11*a22 - a12*a21
        mapped = [apply_one(L, A) for L in loops]
        if fix_winding and det < 0.0:
            mapped = _enforce_outer_ccw(mapped[::-1]) if False else _enforce_outer_ccw(mapped)
        else:
            mapped = _enforce_outer_ccw(mapped)
        out[pid] = mapped
    return out

def piece_ids_from_gradegroup(grade_group, fallback_piece_ids):
    """优先用 GradeGroup 中出现过矩阵的片；没有则回退服装对象里的 clothPieces"""
    ids = [int(pid) for pid,_ in (grade_group.get("clothPieceFabricBaseMatrix") or [])]
    return ids if ids else [int(x) for x in (fallback_piece_ids or [])]

def build_layout_affine(grade_group):
    layout_affine = {}
    for pid, m16 in (grade_group.get("clothPieceFabricBaseMatrix") or []):
        layout_affine[int(pid)] = mat4_to_affine2d(m16)
    return layout_affine



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_json")
    ap.add_argument("-o","--outdir", default="out_style3d_svg")
    ap.add_argument("--variant", choices=["cut","with_seam","both","all"], default="both",
                    help="导出哪种：cut 仅裁线；with_seam 含缝份；both 两张；all 还加 seam_band。")
    ap.add_argument("--place", action="store_true", help="使用 ClothPieceGradeInfo.matrix3D 还原项目摆放")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_json(args.project_json)
    all_classes, by_id = build_indexes(data)

    garments = all_classes.get(4038497362, [])
    if not garments:
        print("[ERR] 找不到服装对象"); return
    G = garments[0]
    
    current_name = (data.get("_fileName","proj").split("~")[0] or "proj").strip()
    grade_group = find_grade_group(all_classes)

    # ---- 基于 GradeGroup 的新流程 ----
    grade_ids = list(grade_group.get("grades") or [])
    layout_affine = build_layout_affine(grade_group)
    base_grade_id = grade_group.get("baseGrade")
    base_grade = by_id.get(int(base_grade_id)) if base_grade_id is not None else None
    base_grade_delta = base_grade.get("deltas") if base_grade else None
    base_grade_curve_delta = base_grade.get("curveVertexDeltas") if base_grade else None
    print(f"[INFO] {len(base_grade_delta)} vertex deltas in base grade {len(base_grade_curve_delta)} curve deltas")
    base_grade_point = [pair[0] for pair in base_grade_delta] if base_grade_delta else []
    base_grade_curve_point = [pair[0] for pair in base_grade_curve_delta] if base_grade_curve_delta else []
    print(f"[INFO] found base_grade_pint count={len(base_grade_point)}, curve_point_count={len(base_grade_curve_point)}")    
    # 片清单：优先 GradeGroup；否则回退服装对象 G['clothPieces']
    fallback_piece_ids = G.get("clothPieces", [])
    cloth_piece_ids_global = piece_ids_from_gradegroup(grade_group, fallback_piece_ids)

    for gid in grade_ids:
        grade_obj = by_id.get(int(gid))
        if not grade_obj:
            continue
        size_name = grade_obj.get("_name", f"G{gid}")

        # ------- 关键：每个尺码优先使用自己的 clothPieceInfoMap -------
        piece_ids_this = [int(p[0]) for p in (grade_obj.get("clothPieceInfoMap") or [])]
        piece_ids_this = piece_ids_this and fallback_piece_ids
        if not piece_ids_this:
            piece_ids_this = cloth_piece_ids_global

        piece_ids_this = [2728]
        # 以“grade 节点”驱动生成当前尺码的裁线 / 缝线
        res = build_loops_for_size(by_id, grade_obj, piece_ids_this)

        cut_by_piece         = {pid: v["cut"]         for pid, v in res.items()}
        seamline_in_by_piece = {pid: v["seamline_in"] for pid, v in res.items()}
        with_seam_by_piece   = {pid: v["with_seam"]   for pid, v in res.items()}
        seam_band_by_piece   = {pid: v["seam_band"]   for pid, v in res.items() if v["seam_band"]}

        # 调试：粗裁线 + 细缝线（虚线）
        out_dbg = os.path.join(args.outdir, f"{current_name}_{size_name}_debug_cut_vs_seam.svg")
        render_cut_and_seamline(cut_by_piece, seamline_in_by_piece, out_dbg)
        from size_to_svg_sym import render_cut_innerfill
        render_cut_innerfill(cut_by_piece, seamline_in_by_piece, out_dbg, color="#0A2A6B")

        # 4) 输出
        if args.variant in ("cut","both","all"):
            out_cut = os.path.join(args.outdir, f"{current_name}_{size_name}_cut.svg")
            from size_to_svg_sym import render_filled_cut
            render_filled_cut(cut_by_piece, out_cut, color="#0A2A6B")

        if args.variant in ("with_seam","both","all"):
            out_ws = os.path.join(args.outdir, f"{current_name}_{size_name}_withSeam.svg")
            render_combined_variant(with_seam_by_piece, out_ws)

        if args.variant == "all" and seam_band_by_piece:
            out_sb = os.path.join(args.outdir, f"{current_name}_{size_name}_seamBand.svg")
            render_combined_variant(seam_band_by_piece, out_sb)

        print(f"[OK] grade {size_name} ({gid}) done.")

if __name__ == "__main__":
    main()
    