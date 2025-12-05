#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DXF (multi-size) -> per-size SVG + per-size spec.json  [robust-encoding edition]

- 读取 DXF 时优先 ezdxf；失败回退 ASCII 解析
- 依据 $DWGCODEPAGE 猜编码，并对块名进行 mojibake 修复
- 严格的尺码 token 匹配（避免 S 命中 XS/XXS）
- SVG 深蓝填充；spec.json 含 vertices/edges（按角点分段）

Usage:
  python dxf_sizes_to_svg_and_spec_v2.py input.dxf -o out_dir
  python dxf_sizes_to_svg_and_spec_v2.py input.dxf -o out_dir --sizes "XXS,XS,S,M,L,XL,XXL"
  python dxf_sizes_to_svg_and_spec_v2.py input.dxf -o out_dir --no-svg-text
"""
import os, re, sys, math, json, argparse, glob

# ---------- 代码页辅助 ----------
CODEPAGE_MAP = {
    "ANSI_936": "gbk", "GBK": "gbk", "GB2312": "gbk", "GB18030": "gb18030",
    "ANSI_1252":"cp1252", "ANSI_932":"cp932", "ANSI_950":"cp950",
    "UTF-8":"utf-8", "UTF8":"utf-8"
}

def tolerant_read_text(path: str) -> str:
    raw = open(path, "rb").read()
    for enc in ("utf-8","gb18030","utf-16","latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")

def read_text_guess(path, force_encoding=None):
    raw = open(path, "rb").read()
    if force_encoding:
        return raw.decode(force_encoding, errors="replace"), force_encoding
    head = raw[:4096].decode("latin-1", errors="ignore")
    m = re.search(r"\$DWGCODEPAGE\s*9\s*'?\s*([A-Za-z0-9_+-]+)", head)
    enc = CODEPAGE_MAP.get(m.group(1).upper(), None) if m else None
    tried = [enc, "utf-8", "gb18030", "gbk", "cp1252", "latin-1"]
    for e in tried:
        if not e: 
            continue
        try:
            return raw.decode(e), e
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace"), "utf-8"

def repair_mojibake(s: str, target="gb18030"):
    """把 'Ã¦Ã¡...' 这类 latin-1 误解码串尽量还原"""
    try:
        return s.encode("latin-1", errors="strict").decode(target, errors="strict")
    except Exception:
        return s

def smart_fix_block_name(name: str, target_codepage: str | None):
    """
    参照 dxf2spec_hmm_strict 的容错思路：
    - 若包含 'Ã'、'�'、高位符号且像被 latin-1 误解码，尝试按目标 codepage 还原
    - 多路兜底：gb18030 / gbk / cp1252
    """
    suspicious = ("Ã" in name) or ("�" in name) or bool(re.search(r"[\x80-\xFF]", name))
    if not suspicious:
        return name
    tried = []
    if target_codepage:
        tried.append(target_codepage)
    tried += ["gb18030", "gbk", "cp1252"]
    for cp in tried:
        fixed = repair_mojibake(name, target=cp)
        # 如果修复后出现明显的中英文且不再含大量 'Ã'，认为成功
        if ("Ã" not in fixed) and (fixed != name):
            return fixed
    return name

# ---------- 名称英文化 ----------
ZH2EN = [
    (r"(前|前片|前身|\(front\))", "front"),
    (r"(后|后片|后身|背|\(back\))", "back"),
    (r"(袖|袖子|袖片)", "sleeve"),
    (r"(领|领子|领片|罗纹|领圈)", "collar"),
    (r"(袖口|罗纹袖口)", "cuff"),
    (r"(下摆|摆条|下摆条|下口|下缝条)", "hem_band"),
    (r"(侧|侧片)", "side_panel"),
    (r"(口袋|袋布)", "pocket"),
    (r"(门襟|门襟条)", "placket"),
    (r"(腰|腰头|腰条|腰带)", "waistband"),
    (r"(裤|裤片|裤腿)", "pant"),
    (r"(裙身|裙片)", "skirt_panel"),
]

def _slugify(s: str) -> str:
    s = re.sub(r"[-_ ]?(xxs|xs|s|m|l|xl|xxl|2xs|2xl|3xl|4xl|5xl)\b", "", s, flags=re.I)
    s = re.sub(r"[^\w]+", "_", s).strip("_").lower()
    return s or "panel"

def to_english_name(s):
    t = s
    for pat, repl in ZH2EN:
        if re.search(pat, t, flags=re.I):
            t = re.sub(pat, repl, t, flags=re.I)
    return _slugify(t)

def guess_label_from_en(name_en):
    if "front" in name_en: return "front"
    if "back" in name_en:  return "back"
    if "sleeve" in name_en:return "sleeve"
    if "collar" in name_en:return "collar"
    if "cuff" in name_en:   return "cuff"
    if "placket" in name_en:return "placket"
    if "hem" in name_en or "band" in name_en: return "hem_band"
    return "panel"

# ---------- 多路 DXF 解析 ----------
def polygon_area(pts):
    a=0.0
    for (x1,y1),(x2,y2) in zip(pts, pts[1:]): a += x1*y2 - x2*y1
    return 0.5*a

def parse_blocks_loops_ascii(dxf_text):
    lines=[ln.rstrip("\r\n") for ln in dxf_text.splitlines()]
    n=len(lines); i=0; panels={}
    while i<n-1:
        if lines[i].strip()=="0" and lines[i+1].strip()=="BLOCK":
            i+=2; name=None; loops=[]
            while i<n-1:
                c=lines[i].strip(); v=lines[i+1].strip()
                if c=="2" and name is None:
                    name=v; i+=2; continue
                if c=="0" and v=="ENDBLK":
                    if loops:
                        loops=sorted(loops, key=lambda p: abs(polygon_area(p)), reverse=True)
                        panels[name]=loops
                    i+=2; break
                if c=="0" and v=="LWPOLYLINE":
                    i+=2; pts=[]; closed=0
                    while i<n-1:
                        cc=lines[i].strip(); vv=lines[i+1].strip()
                        if cc=="70":
                            try: closed=int(vv)
                            except: closed=0
                            i+=2; continue
                        if cc=="10":
                            try:
                                x=float(v); y=float(lines[i+3].strip()) if i+3<n and lines[i+2].strip()=="20" else None
                            except: x=y=None
                            i+=4 if y is not None else 2
                            if x is not None and y is not None: pts.append((x,y))
                            continue
                        if cc=="0":
                            if (closed&1) and len(pts)>=3:
                                if pts[0]!=pts[-1]: pts.append(pts[0])
                                loops.append(pts)
                            break
                        i+=2
                    continue
                if c=="0" and v=="POLYLINE":
                    i+=2; pts=[]; closed=0
                    while i<n-1:
                        cc=lines[i].strip(); vv=lines[i+1].strip()
                        if cc=="70":
                            try: closed=int(vv)
                            except: closed=0
                            i+=2; continue
                        if cc=="0" and vv=="VERTEX":
                            i+=2; x=y=None
                            while i<n-1 and lines[i].strip()!="0":
                                c3=lines[i].strip(); v3=lines[i+1].strip()
                                if c3=="10": x=float(v3)
                                if c3=="20": y=float(v3)
                                i+=2
                            if x is not None and y is not None: pts.append((x,y))
                            continue
                        if cc=="0" and vv=="SEQEND":
                            if (closed&1) and len(pts)>=3:
                                if pts[0]!=pts[-1]: pts.append(pts[0])
                                loops.append(pts)
                            i+=2; break
                        i+=2
                    continue
                i+=2
        i+=1
    return panels

def parse_blocks_loops_ezdxf(path:str):
    import ezdxf
    doc = ezdxf.readfile(path)
    panels = {}
    used = set(e.dxf.name for e in doc.modelspace().query("INSERT"))
    for bname in used:
        try:
            blk = doc.blocks[bname]
        except Exception:
            continue
        loops=[]
        for ent in blk:
            try:
                if ent.dxftype()=="LWPOLYLINE":
                    pts=[(p[0],p[1]) for p in ent.get_points()]
                    if ent.closed and len(pts)>=3:
                        if pts[0]!=pts[-1]: pts.append(pts[0])
                        loops.append(pts)
                elif ent.dxftype()=="POLYLINE":
                    pts=[]
                    for v in ent.vertices:
                        pts.append((float(v.dxf.location.x), float(v.dxf.location.y)))
                    if ent.is_closed and len(pts)>=3:
                        if pts[0]!=pts[-1]: pts.append(pts[0])
                        loops.append(pts)
            except Exception:
                continue
        if loops:
            panels[bname]=sorted(loops, key=lambda p: abs(polygon_area(p)), reverse=True)
    # 尝试取 header 的 codepage，供后续修名
    codepage = None
    try:
        codepage = doc.header.get("$DWGCODEPAGE", None)
    except Exception:
        pass
    return panels, CODEPAGE_MAP.get((codepage or "").upper(), None)

# ---------- 尺码严格匹配 ----------
SIZE_EQUIV = {
    "2XS":["2XS","XXS"], "XXS":["XXS","2XS"], "XS":["XS"], "S":["S"], "M":["M"],
    "L":["L"], "XL":["XL"], "2XL":["2XL","XXL"], "XXL":["XXL","2XL"],
    "3XL":["3XL"], "4XL":["4XL"], "5XL":["5XL"]
}
def canonical_size_token(s):
    s=(s or "").upper().replace(" ","").replace("_","").replace("-","")
    for k,alts in SIZE_EQUIV.items():
        if s in alts: return k
    return s
def strict_suffix_regex(size_token):
    can = canonical_size_token(size_token)
    alts = SIZE_EQUIV.get(can, [can])
    alts_l=[a.lower() for a in alts]
    return re.compile(r"(?:^|[^A-Za-z0-9])(?:" + "|".join(re.escape(a) for a in alts_l) + r")(?:$|[^A-Za-z0-9])", re.I)

def detect_sizes_from_names(names:list[str]) -> list[str]:
    found=set()
    for nm in names:
        for tok in re.split(r"[^A-Za-z0-9]+",(nm or "")):
            up = tok.upper()
            for k,alts in SIZE_EQUIV.items():
                if up in alts:
                    found.add(k)
    order=["XXS","XS","S","M","L","XL","XXL"]
    return [s for s in order if s in found]

# ---------- 角点与边 ----------
def detect_corners(pts, thresh_deg=160.0, min_gap=2):
    def ang(v1,v2):
        a1=math.atan2(v1[1],v1[0]); a2=math.atan2(v2[1],v2[0])
        return abs((a2-a1+math.pi)%(2*math.pi)-math.pi)*180.0/math.pi
    idx=[]
    for i in range(1,len(pts)-1):
        v1=(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
        v2=(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
        if abs(v1[0])+abs(v1[1])<1e-12 or abs(v2[0])+abs(v2[1])<1e-12: continue
        if ang(v1,v2) < thresh_deg and (not idx or i-idx[-1]>=min_gap): idx.append(i)
    return [0]+idx+[len(pts)-1]

def make_vertices_edges(contour, corner_deg=160.0):
    if contour[0]!=contour[-1]: contour=contour+[contour[0]]
    idxs = detect_corners(contour, thresh_deg=corner_deg, min_gap=2)
    verts = [[float(contour[i][0]), float(contour[i][1])] for i in idxs[:-1]]
    n=len(verts)
    if n<3:
        n=4; L=len(contour)-1
        verts=[[float(contour[int(i*L/n)][0]), float(contour[int(i*L/n)][1])] for i in range(n)]
    edges=[{"endpoints":[i,(i+1)%n]} for i in range(n)]
    return verts, edges

# ---------- SVG ----------
def render_svg(path, pieces, fill="#0A2A6B", op=0.22, add_text=True):
    xs=[]; ys=[]
    for P in pieces:
        xs += [p[0] for p in P["contour"]]
        ys += [p[1] for p in P["contour"]]
    if not xs:
        open(path,"w").write("<svg xmlns='http://www.w3.org/2000/svg'/>"); return
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
    pad=50; w=maxx-minx+2*pad; h=maxy-miny+2*pad
    def TR(p): return (p[0]-minx+pad, (maxy-p[1])+pad)
    out=[f"<svg xmlns='http://www.w3.org/2000/svg' width='{w:.0f}' height='{h:.0f}' viewBox='0 0 {w:.0f} {h:.0f}'>"]
    for P in pieces:
        pts=[TR(p) for p in P["contour"]]
        d="M "+" L ".join(f"{x:.1f} {y:.1f}" for x,y in pts)+" Z"
        out.append(f"<path d='{d}' fill='{fill}' fill-opacity='{op}' stroke='#222' stroke-width='1.2'/>")
        if add_text:
            cx=sum(x for x,y in pts[:-1])/max(1,len(pts)-1); cy=sum(y for x,y in pts[:-1])/max(1,len(pts)-1)
            out.append(f"<text x='{cx:.1f}' y='{cy:.1f}' font-size='14' fill='#ffffffaa' text-anchor='middle'>{P['name_en']}</text>")
    out.append("</svg>")
    open(path,"w",encoding="utf-8").write("\n".join(out))

# === 轻量“翻译器”钩子（可外接本地小模型 HTTP 服务） ===
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

_ZH2EN_RULES = [
    (r"(前|前片|前身)", "front"),
    (r"(后|后片|后身|背)", "back"),
    (r"(袖|袖子|袖片)", "sleeve"),
    (r"(领|领子|领片|罗纹|领圈)", "collar"),
    (r"(袖口)", "cuff"),
    (r"(下摆|摆条|下口|下摆条)", "hem_band"),
    (r"(门襟|门襟条)", "placket"),
    (r"(口袋|袋布)", "pocket"),
]
def _rule_translate(s: str) -> str:
    t = s
    for pat, repl in _ZH2EN_RULES:
        t = re.sub(pat, repl, t, flags=re.I)
    return _slugify(t)

def _http_post_json(url, payload, timeout=6.0):
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type":"application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

class Translator:
    def __init__(self, trans_api: str | None, cache_path: str | None):
        self.trans_api = trans_api
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, "r", encoding="utf-8"))
            except Exception:
                self.cache = {}

    def translate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "panel"
        if text in self.cache:
            return self.cache[text]
        best = _rule_translate(text)
        if self.trans_api:
            try:
                cand = (_http_post_json(self.trans_api, {"text": text}) or {}).get("text","").strip()
                if cand:
                    best = _slugify(cand)
            except (URLError, HTTPError, TimeoutError, ValueError):
                pass
            except Exception:
                pass
        self.cache[text] = best
        if self.cache_path:
            try:
                json.dump(self.cache, open(self.cache_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
            except Exception:
                pass
        return best


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

# ---------- 主流程 ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dxf")
    ap.add_argument("-o","--outdir", default="out_sizes")
    ap.add_argument("--sizes", help="XXS,XS,S,M,L,XL,XXL ...")
    ap.add_argument("--corner-deg", type=float, default=160.0)
    ap.add_argument("--units", choices=["m","cm","mm","inch"], default="mm")
    ap.add_argument("--no-svg-text", action="store_true", help="SVG 中不绘制文字")
    ap.add_argument("--force-encoding", help="强制按该编码读取 DXF（如 gb18030/gbk/cp936）")
    ap.add_argument("--trans-api", default=None,
                    help="翻译服务HTTP地址（POST {'text':'中文名'} -> {'text':'英文名'}）")
    ap.add_argument("--trans-cache", default="trans_cache.json",
                    help="翻译缓存 JSON，默认 trans_cache.json")
    args=ap.parse_args()

    # translator = Translator(args.trans_api, args.trans_cache)
    units={"m":1.0,"cm":0.01,"mm":0.001,"inch":0.0254}[args.units]
    all_sizes = {'XXS':'XS','2XS':'XS','XS':'S','S':'M','M':'L','L':'XL','XL':'XXL','XXL':'XXXL','2XL':'XXXL','3XL':'4XL','4XL':'5XL'}
    os.makedirs(args.outdir, exist_ok=True)
    garment_num = 0

    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()
    
    for f in glob.glob(os.path.join(args.dxf, "*.dxf")):
        garment_num += 1
        base=os.path.splitext(os.path.basename(f))[0]

        # 读取原文 & 头部编码猜测
        text, used_enc = read_text_guess(f, force_encoding=args.force_encoding)

        # 优先 ezdxf，拿不到再回退 ASCII
        codepage_target = None
        blocks = {}
        try:
            blocks, codepage_target = parse_blocks_loops_ezdxf(f)
            used = "ezdxf"
        except Exception:
            blocks = parse_blocks_loops_ascii(text)
            used = "ascii"
        if not blocks:
            print("[ERR] Failed to parse BLOCK/LWPOLYLINE."); sys.exit(2)

        # 修复块名乱码并提取尺码
        fixed_blocks={}
        for raw_name, loops in blocks.items():
            name = smart_fix_block_name(raw_name, codepage_target or CODEPAGE_MAP.get((used_enc or "").upper(), None))
            fixed_blocks[name] = loops

        detected = detect_sizes_from_names(list(fixed_blocks.keys()))
        sizes = [canonical_size_token(s.strip()) for s in args.sizes.split(",")] if args.sizes else detected
        if not sizes:
            print("[ERR] no sizes detected"); sys.exit(3)
        print(f"[*] parser={used}, codepage={used_enc or codepage_target}, sizes={sizes}")

        units_in_meter=float(units)
        sizes_first = 0
        base_en = translate_text(translate_client, base)[0]['translatedText']
        names_en = {}
        # pre_names_en = {}
        for sz in sizes:
            rx = strict_suffix_regex(sz)
            pieces=[]
            
            for nm,loops in fixed_blocks.items():
                if not rx.search((nm or "").lower()):
                    continue
                loop=loops[0]
                if polygon_area(loop)<0: 
                    loop=loop[::-1]
                if sizes_first==0:
                    name_en = translate_text(translate_client, nm.strip().split("-")[0])[0]['translatedText']
                    names_en[nm.strip().split("-")[0]] = name_en.strip().split("-")[0]
                    # pre_names_en[nm.strip().split("-")[0]] = name_en
                
                # print(nm, names_en[nm.strip().split("-")[0]])
                pieces.append({"name_raw":nm, "name_en":names_en[nm.strip().split("-")[0]], "contour":loop})
            # print(names_en)
            # print(pre_names_en)
            sizes_first += 1
            
            if not pieces:
                print(f"[WARN] size {sz} has no panels"); 
                continue

            # spec（英文）
            panels={}; order=[]
            for P in pieces:
                verts,edges = make_vertices_edges(P["contour"], corner_deg=args.corner_deg)
                label = P["name_en"]
                # new_name_and_sz = f"{label}_{all_sizes.get(sz, sz)}"
                new_name_and_sz = f"{label}_{sz}"
                panels[new_name_and_sz] = {
                    "translation":[0,0], "rotation":0, "label":label,
                    "vertices":verts, "edges":edges,
                    "meta":{"raw_name":P["name_raw"]},
                    "contours":{"main":[[float(x),float(y)] for x,y in P["contour"]]}
                }
                order.append(P["name_en"])

            spec = {
                "properties":{
                    "units_in_meter": units_in_meter,
                    "normalized_edge_loops": False,
                    "normalize_panel_translation": False,
                    "curvature_coords":"relative"
                },
                "pattern":{"panels":panels, "stitches":[], "panel_order":order},
                "parameters":{}, "parameter_order":[]
            }
            
            new_base_en = re.sub(r"[^\w]+", "_", base_en).strip("_").lower() or "garment"
            # if len(sizes) < 7:
            #     new_sz = sz
            # else:
            #     new_sz = all_sizes.get(sz, sz)
            # out_dir=os.path.join(args.outdir, f"{new_base_en}_{new_sz}")
            out_dir=os.path.join(args.outdir, f"{new_base_en}_{sz}")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir,"spec.json"),"w",encoding="utf-8") as f_n:
                json.dump(spec, f_n, ensure_ascii=False, indent=2)

            render_svg(os.path.join(out_dir,"pattern.svg"), pieces,
                    fill="#0A2A6B", op=0.22, add_text=(not args.no_svg_text))
            print(f"[OK] {sz} -> {out_dir}")
        print(f"[ok] total sizes processed for {base}: {len(sizes)}")
    print(f"[OK] {garment_num}")
        

if __name__=="__main__":
    main()
