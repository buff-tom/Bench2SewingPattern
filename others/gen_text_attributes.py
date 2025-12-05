import os
import json
import re
import time
import warnings
import typing_extensions
from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError, NotFound, PermissionDenied

# ================= 2. 配置区域（修改这里！）=================
API_KEY = os.getenv("GOOGLE_API_KEY")
TARGET_FOLDER = "/home/user/buff-tomma/Bench2SewingPattern_DataSet/female/female_asia_model"

# ================= 3. 初始化Gemini Client =================
client = genai.Client(api_key=API_KEY)

# ================= 4. 人体数据（保留原有） =================
BODY_SPECS = {
    "A_Female": {
        "XXS": {"height": 153, "bust": 72, "waist": 54, "hip": 78},
        "XS":  {"height": 156, "bust": 76, "waist": 58, "hip": 82},
        "S":   {"height": 159, "bust": 80, "waist": 62, "hip": 86},
        "M":   {"height": 162, "bust": 84, "waist": 66, "hip": 90},
        "L":   {"height": 165, "bust": 88, "waist": 70, "hip": 94},
        "XL":  {"height": 168, "bust": 92, "waist": 74, "hip": 98},
        "XXL": {"height": 171, "bust": 96, "waist": 78, "hip": 102},
        "2XL": {"height": 171, "bust": 96, "waist": 78, "hip": 102},
    },
    "E_Female": {
        "XXS": {"height": 159, "bust": 80, "waist": 62, "hip": 86},
        "XS":  {"height": 162, "bust": 84, "waist": 66, "hip": 90},
        "S":   {"height": 165, "bust": 88, "waist": 70, "hip": 94},
        "M":   {"height": 168, "bust": 92, "waist": 74, "hip": 98},
        "L":   {"height": 171, "bust": 96, "waist": 78, "hip": 102},
        "XL":  {"height": 174, "bust": 100, "waist": 82, "hip": 106},
        "XXL": {"height": 177, "bust": 104, "waist": 86, "hip": 110},
        "2XL": {"height": 177, "bust": 104, "waist": 86, "hip": 110},
    },
    "A_Male": {
        "XS":  {"height": 169, "bust": 84, "waist": 70, "hip": 82},
        "S":   {"height": 172, "bust": 88, "waist": 74, "hip": 86},
        "M":   {"height": 175, "bust": 92, "waist": 78, "hip": 90},
        "L":   {"height": 178, "bust": 96, "waist": 82, "hip": 94},
        "XL":  {"height": 181, "bust": 100, "waist": 86, "hip": 98},
        "2XL": {"height": 184, "bust": 104, "waist": 90, "hip": 102},
        "3XL": {"height": 187, "bust": 108, "waist": 94, "hip": 106},
    },
    "E_Male": {
        "XS":  {"height": 175, "bust": 92, "waist": 78, "hip": 90},
        "S":   {"height": 178, "bust": 96, "waist": 82, "hip": 94},
        "M":   {"height": 181, "bust": 100, "waist": 86, "hip": 98},
        "L":   {"height": 184, "bust": 104, "waist": 90, "hip": 102},
        "XL":  {"height": 187, "bust": 108, "waist": 94, "hip": 106},
        "2XL": {"height": 190, "bust": 112, "waist": 98, "hip": 110},
        "3XL": {"height": 193, "bust": 116, "waist": 102, "hip": 114},
    }
}


def parse_gemini_json(raw_text: str):
    """
    从 Gemini 输出中提取 JSON：
    - 去掉 ```json ... ``` 或 ``` ... ``` 代码块包裹
    - 再用 json.loads 解析
    """
    if not raw_text:
        raise json.JSONDecodeError("Empty response", "", 0)

    s = raw_text.strip()

    # 1) 去掉 ```json ... ``` 包裹
    if s.startswith("```"):
        # 形如 ```json\n{...}\n``` 或 ```\n{...}\n```
        # 先去掉开头的 ```json 或 ```
        s = s.lstrip("`")
        # 去掉可能的 "json" 前缀
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        # 再去掉结尾的 ```
        if s.endswith("```"):
            s = s[:-3].strip()

    # 2) 现在 s 应该就是形如 { ... } 的字符串了
    return json.loads(s)


# ================= 5. 定义Schema =================
class SizeAwareDescription(typing_extensions.TypedDict, total=False):
    garment_category: str
    target_gender: str
    target_region: str
    size: str
    model_body_specs: dict[str, int]
    fit_analysis: str
    visual_description: str
    prompt_for_generation: str
    # 组合后的描述：visual + fit，作为主描述给 benchmark 用
    description: str
    original_file: list[str]


# ================= 6. 核心函数 =================
def get_body_specs(gender_key, region_key, size_label):
    dict_key = f"{region_key}_{gender_key}"
    specs = BODY_SPECS.get(dict_key, {})
    size_upper = size_label.upper()
    return specs.get(size_upper, None)

def analyze_model_image(image_path, category_name, size, specs, gender, region, max_retries=3):
    specs_str = "Unknown"
    if specs:
        specs_str = (
            f"Height: {specs['height']}cm, Bust: {specs['bust']}cm, "
            f"Waist: {specs['waist']}cm, Hip: {specs['hip']}cm"
        )

    # 修改点 1: 优化 Prompt，强调这是 3D 模特/假人，减少被误判为真人的概率
    prompt = f"""
    You are a Technical Fashion Designer analyzing a fit session on a 3D digital avatar.
    The image contains a standard 3D fashion mannequin/avatar used solely for garment simulation.
    
    Your output MUST be a valid JSON string that strictly conforms to the following schema:
    {{
      "garment_category": string,
      "target_gender": string,
      "target_region": string,
      "size": string,
      "model_body_specs": {json.dumps(specs) if specs else "{}"},
      "fit_analysis": string,
      "visual_description": string,
      "prompt_for_generation": string
    }}
    Do NOT add any extra text, only the JSON string.

    Context:
    - Garment: {category_name}
    - Size: {size}
    - Target Demographic: {region} {gender}
    - Model Body Measurements: {specs_str}

    Task:
    1. Fit Analysis: Analyze ease at bust/chest, waist, and hips. State clearly whether the garment appears tight, regular, or loose.
    2. Visual Description: Describe the visual appearance of the garment (sleeves, length, collar, etc.) as it appears on the model.
    3. Prompt for Generation: Create a high-quality text prompt that includes size, body context, and garment style, suitable for a Text-to-Image model.
    """

    for retry in range(max_retries):
        try:
            print(f"   -> Sending to Gemini 2.5 Flash... (Size: {size}, Retry: {retry+1})")
            
            uploaded_file = client.files.upload(file=image_path[0])
            # 读取第二张图（如果有）作为 bytes
            with open(image_path[1], "rb") as f:
                img2_bytes = f.read()
            
            # 根据文件后缀判断 mime_type
            mime_type_local = "image/jpeg" if image_path[1].lower().endswith((".jpg", ".jpeg")) else "image/png"

            contents = [
                prompt.strip(),
                uploaded_file,
                types.Part.from_bytes(data=img2_bytes, mime_type=mime_type_local),
            ]

            # 修改点 2: 添加 config 并配置 safety_settings
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE"
                        ),
                    ]
                )
            )

            # 检查是否因为安全原因被拦截但没报错（返回空文本）
            if not response.text:
                reason = "Unknown"
                if response.candidates:
                    reason = response.candidates[0].finish_reason
                print(f"   [Warning] Empty Response. Finish Reason: {reason}")
                
                # 如果明确是 SAFETY 原因，即使重试可能也没用，但在 BLOCK_NONE 下应该极少出现
                if str(reason) == "SAFETY":
                    print("   [Tip] Even with BLOCK_NONE, Google may block hard-core NSFW. Ensure prompt emphasizes '3D mannequin'.")
                    client.files.delete(name=uploaded_file.name)
                    return None
                
                # 其他原因则抛出异常触发重试
                raise ValueError("Response text is empty.")

            text = response.text
            data: SizeAwareDescription = parse_gemini_json(text)

            if not data.get("model_body_specs") and specs:
                data["model_body_specs"] = specs

            visual = (data.get("visual_description") or "").strip()
            fit = (data.get("fit_analysis") or "").strip()
            if visual or fit:
                parts = []
                if visual: parts.append(visual)
                if fit: parts.append(f"Fit Analysis: {fit}")
                data["description"] = " ".join(parts)

            client.files.delete(name=uploaded_file.name)
            return data

        except PermissionDenied as e:
            print(f"   [Error] 权限拒绝/地区限制: {e}")
            break
        except NotFound as e:
            print(f"   [Error] 模型/文件不存在: {e}")
            break
        except Exception as e:
            print(f"   [Error] {e}")
            # 尝试清理上传的文件
            try: client.files.delete(name=uploaded_file.name)
            except: pass
            
            if retry < max_retries - 1:
                time.sleep(2 * (retry + 1))
            else:
                print(f"   [Failed] Analysis failed after retries.")

    return None


# ================= 7. 遍历处理+入口 =================
def process_root_folder(root_path: str):
    root_name = os.path.basename(root_path).lower()

    if "female" in root_name:
        gender = "Female"
    elif "male" in root_name:
        gender = "Male"
    else:
        gender = "Unknown"

    if "eur" in root_name:
        region = "Eur"
    elif "asia" in root_name or "亚" in root_name:
        region = "Asia"
    else:
        region = "Asia"

    print(f"🚀 Processing Root: {root_name} | Gender: {gender} | Region: {region}")

    # 按款式目录遍历
    for category_dir in os.listdir(root_path):
        cat_path = os.path.join(root_path, category_dir)
        if not os.path.isdir(cat_path):
            continue

        for size_dir in os.listdir(cat_path):
            size_path = os.path.join(cat_path, size_dir)
            if not os.path.isdir(size_path):
                continue
            
            if "description.json" in os.listdir(size_path):
                print(f"   ⏭️ Skip existing description: {category_dir}/{size_dir}")
                continue
            # 查找该 size 下的图片
            image_candidates = [
                f
                for f in os.listdir(size_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if len(image_candidates) < 2:
                continue

            image_candidates.sort()
            target_image = [
                os.path.join(size_path, image_candidates[0]),
                os.path.join(size_path, image_candidates[1]),
            ]

            body_specs = get_body_specs(gender, region, size_dir)

            data = analyze_model_image(
                target_image,
                category_name=category_dir,
                size=size_dir,
                specs=body_specs,
                gender=gender,
                region=region,
            )

            if data:
                # original_file：使用与 JSON 同目录下的文件名
                basenames = [os.path.basename(p) for p in target_image]
                data["original_file"] = basenames

                save_path = os.path.join(size_path, "description.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"   ✅ Saved description for {category_dir}/{size_dir} -> {save_path}")

            time.sleep(1)


if __name__ == "__main__":
    if os.path.exists(TARGET_FOLDER):
        process_root_folder(TARGET_FOLDER)
    else:
        print(f"Folder not found: {TARGET_FOLDER}")
