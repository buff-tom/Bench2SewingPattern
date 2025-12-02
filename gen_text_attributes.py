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
API_KEY = "*"  # 替换为真实Key
TARGET_FOLDER = "/home/user/buff-tomma/Pattern_Making/male_garment/male_asia_front_and_back_garment_with_model"

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

# ================= 5. 定义Schema（保留原有） =================
class SizeAwareDescription(typing_extensions.TypedDict):
    garment_category: str
    target_gender: str
    target_region: str
    size: str
    model_body_specs: dict[str, int]
    fit_analysis: str
    visual_description: str
    prompt_for_generation: str

# ================= 6. 核心函数（保留原有） =================
def get_body_specs(gender_key, region_key, size_label):
    dict_key = f"{region_key}_{gender_key}"
    specs = BODY_SPECS.get(dict_key, {})
    size_upper = size_label.upper()
    return specs.get(size_upper, None)

def analyze_model_image(image_path, category_name, size, specs, gender, region, max_retries=3):
    specs_str = "Unknown"
    if specs:
        specs_str = f"Height: {specs['height']}cm, Bust: {specs['bust']}cm, Waist: {specs['waist']}cm, Hip: {specs['hip']}cm"

    prompt = f"""
    You are a Technical Fashion Designer analyzing a fit session.
    Your output MUST be a valid JSON string that strictly conforms to the following schema:
    {{
        "garment_category": string,
        "target_gender": string,
        "target_region": string,
        "size": string,
        "model_body_specs": {json.dumps(specs) if specs else "{}"},
        "fit_analysis": string (analyze ease at bust/chest, waist, hips; tight/regular/loose),
        "visual_description": string (describe sleeves, length, collar, etc.),
        "prompt_for_generation": string (high-quality prompt for Text-to-Image)
    }}
    Do NOT add any extra text, only the JSON string.

    **Context:**
    - **Garment:** {category_name}
    - **Size:** {size}
    - **Target Demographic:** {region} {gender}
    - **Model Body Measurements:** {specs_str}
    
    **Task:**
    1. **Fit Analysis:** Analyze how this specific size fits this specific body. Mention ease (looseness) at the bust/chest, waist, and hips. Does it look tight, regular, or loose?
    2. **Visual Description:** Describe the visual appearance of the garment (sleeves, length, collar) as it appears on the model.
    3. **Prompt for Generation:** Create a high-quality text prompt that includes the size and body context, suitable for a Text-to-Image model to recreate this specific look.
    """
    

    for retry in range(max_retries):
        try:
            print(f"   -> Sending to Gemini 2.5 Flash... (Size: {size}, Retry: {retry+1})")
            uploaded_file = client.files.upload(file=image_path[0])
            with open(image_path[1], 'rb') as f:
                img2_bytes = f.read()
            contents = [prompt.strip(), uploaded_file, 
                        types.Part.from_bytes(
                            data=img2_bytes,
                            mime_type='image/png')
                        ]
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            
            response.resolve()
            result = json.loads(response.text)
            
            if not result.get("model_body_specs") and specs:
                result["model_body_specs"] = specs
            
            client.files.delete(uploaded_file.name)
            return result
        
        except PermissionDenied as e:
            print(f"   [Error] 权限拒绝/地区限制: {e}")
            print("   建议：1. 检查API密钥有效性 2. 确认代理节点在Gemini支持地区（美/日/欧盟）")
            break
        
        except NotFound as e:
            print(f"   [Error] 模型/文件不存在: {e}")
            break
        
        except GoogleAPIError as e:
            print(f"   [Gemini API Error] {e}")
            if retry < max_retries - 1:
                time.sleep(2 * (retry + 1))
            else:
                print(f"   [Failed] 重试{max_retries}次后仍失败")
        
        except json.JSONDecodeError as e:
            print(f"   [Error] JSON解析失败: {e}")
            break
        
        except Exception as e:
            print(f"   [Unknown Error] {e}")
            if retry < max_retries - 1:
                time.sleep(2 * (retry + 1))
    
    return None

# ================= 7. 遍历处理+入口（保留原有） =================
def process_root_folder(root_path):
    root_name = os.path.basename(root_path).lower()
    
    if "female" in root_name: gender = "Female"
    elif "male" in root_name: gender = "Male"
    else: gender = "Unknown"
    
    if "eur" in root_name: region = "Eur"
    elif "asia" in root_name or "亚" in root_name: region = "Asia"
    else: region = "Asia"
    
    print(f"🚀 Processing Root: {root_name} | Gender: {gender} | Region: {region}")
    results = {}
    
    for category_dir in os.listdir(root_path):
        cat_path = os.path.join(root_path, category_dir)
        if not os.path.isdir(cat_path): continue
        
        for size_dir in os.listdir(cat_path):
            size_path = os.path.join(cat_path, size_dir)
            if not os.path.isdir(size_path): continue
            
            image_candidates = [f for f in os.listdir(size_path) if f.lower().endswith(('.png', '.jpg'))]
            if not image_candidates: continue
            target_image = [os.path.join(size_path, image_candidates[0]),os.path.join(size_path, image_candidates[1])]
            
            body_specs = get_body_specs(gender, region, size_dir)
            
            data = analyze_model_image(
                target_image, 
                category_name=category_dir, 
                size=size_dir, 
                specs=body_specs, 
                gender=gender, 
                region=region
            )
            
            if data:
                uid = f"{category_dir}_{size_dir}"
                data["original_file"] = target_image
                results[uid] = data
                
                save_path = os.path.join(cat_path, "size_descriptions.json")
                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            time.sleep(1)

if __name__ == "__main__":
    if os.path.exists(TARGET_FOLDER):
        process_root_folder(TARGET_FOLDER)
    else:
        print(f"Folder not found: {TARGET_FOLDER}")