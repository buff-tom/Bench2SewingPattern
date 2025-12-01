import os
import json
import re
import time
import typing_extensions
import google.generativeai as genai

# ================= 配置区域 =================
API_KEY = "YOUR_GOOGLE_API_KEY"  # 替换你的 Key
DATA_ROOT = "path/to/your/root_folder"  # 例如: buff-tomma/Pattern_Making

# 配置 Gemini
genai.configure(api_key=API_KEY)

# ================= 1. 数字化人体净数据 (来自你的截图) =================
# 数据来源：提供的截图

BODY_SPECS = {
    "Asia_Female": {
        "XXS": {"height": 153, "bust": 72, "waist": 54, "hip": 78},
        "XS":  {"height": 156, "bust": 76, "waist": 58, "hip": 82},
        "S":   {"height": 159, "bust": 80, "waist": 62, "hip": 86},
        "M":   {"height": 162, "bust": 84, "waist": 66, "hip": 90},
        "L":   {"height": 165, "bust": 88, "waist": 70, "hip": 94},
        "XL":  {"height": 168, "bust": 92, "waist": 74, "hip": 98},
        "XXL": {"height": 171, "bust": 96, "waist": 78, "hip": 102}, # 图中对应 2XL
        "2XL": {"height": 171, "bust": 96, "waist": 78, "hip": 102},
    },
    "Eur_Female": {
        "XXS": {"height": 159, "bust": 80, "waist": 62, "hip": 86},
        "XS":  {"height": 162, "bust": 84, "waist": 66, "hip": 90},
        "S":   {"height": 165, "bust": 88, "waist": 70, "hip": 94},
        "M":   {"height": 168, "bust": 92, "waist": 74, "hip": 98},
        "L":   {"height": 171, "bust": 96, "waist": 78, "hip": 102},
        "XL":  {"height": 174, "bust": 100, "waist": 82, "hip": 106},
        "XXL": {"height": 177, "bust": 104, "waist": 86, "hip": 110}, # 图中对应 2XL
        "2XL": {"height": 177, "bust": 104, "waist": 86, "hip": 110},
    },
    "Asia_Male": {
        "XS":  {"height": 169, "bust": 84, "waist": 70, "hip": 82}, # 截图数据
        "S":   {"height": 172, "bust": 88, "waist": 74, "hip": 86},
        "M":   {"height": 175, "bust": 92, "waist": 78, "hip": 90},
        "L":   {"height": 178, "bust": 96, "waist": 82, "hip": 94},
        "XL":  {"height": 181, "bust": 100, "waist": 86, "hip": 98},
        "2XL": {"height": 184, "bust": 104, "waist": 90, "hip": 102},
        "3XL": {"height": 187, "bust": 108, "waist": 94, "hip": 106},
    },
    "Eur_Male": {
        "XS":  {"height": 175, "bust": 92, "waist": 78, "hip": 90},
        "S":   {"height": 178, "bust": 96, "waist": 82, "hip": 94},
        "M":   {"height": 181, "bust": 100, "waist": 86, "hip": 98},
        "L":   {"height": 184, "bust": 104, "waist": 90, "hip": 102},
        "XL":  {"height": 187, "bust": 108, "waist": 94, "hip": 106},
        "2XL": {"height": 190, "bust": 112, "waist": 98, "hip": 110},
        "3XL": {"height": 193, "bust": 116, "waist": 102, "hip": 114},
    }
}

# ================= 2. 定义输出 Schema =================

class SizeAwareDescription(typing_extensions.TypedDict):
    garment_category: str
    target_gender: str
    target_region: str
    size: str
    model_body_specs: dict[str, int]
    fit_analysis: str
    visual_description: str
    prompt_for_generation: str

# 配置 Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": SizeAwareDescription,
        "temperature": 0.2,
    }
)

# ================= 3. 解析与生成逻辑 =================

def get_body_specs(gender_key, region_key, size_label):
    """根据文件名关键字查找对应的人体数据"""
    # 构造 Key，例如 "Asia_Female"
    dict_key = f"{region_key}_{gender_key}"
    
    specs = BODY_SPECS.get(dict_key, {})
    # 尺码归一化 (防止大小写差异)
    size_upper = size_label.upper()
    return specs.get(size_upper, None)

def analyze_model_image(image_path, category_name, size, specs, gender, region):
    """调用 Gemini 生成描述"""
    
    # 构造能够感知数据的 Prompt
    specs_str = "Unknown"
    if specs:
        specs_str = f"Height: {specs['height']}cm, Bust: {specs['bust']}cm, Waist: {specs['waist']}cm, Hip: {specs['hip']}cm"

    prompt = f"""
    You are a Technical Fashion Designer analyzing a fit session.
    
    **Context:**
    - **Garment:** {category_name}
    - **Size:** {size}
    - **Target Demographic:** {region} {gender}
    - **Model Body Measurements:** {specs_str}
    
    **Input:** An image of a 3D model with the specified body measurements wearing the garment.
    
    **Task:**
    1. **Fit Analysis:** Analyze how this specific size fits this specific body. Mention ease (looseness) at the bust/chest, waist, and hips. Does it look tight, regular, or loose?
    2. **Visual Description:** Describe the visual appearance of the garment (sleeves, length, collar) as it appears on the model.
    3. **Prompt for Generation:** Create a high-quality text prompt that includes the size and body context, suitable for a Text-to-Image model to recreate this specific look.
    """

    print(f"   -> Sending to Gemini... (Size: {size})")
    try:
        img_file = genai.upload_file(image_path)
        response = model.generate_content([img_file, prompt])
        return json.loads(response.text)
    except Exception as e:
        print(f"   [Error] {e}")
        return None

def process_root_folder(root_path):
    # 假设 root_path 是 "female_eur_front_garment_with_model" 这一层
    
    root_name = os.path.basename(root_path).lower()
    
    # 1. 自动推断性别和地区
    if "female" in root_name: gender = "Female"
    elif "male" in root_name: gender = "Male"
    else: gender = "Unknown"
    
    if "eur" in root_name: region = "Eur"
    elif "asia" in root_name or "亚" in root_name: region = "Asia"
    else: region = "Asia" # 默认 fallback
    
    print(f"🚀 Processing Root: {root_name} | Gender: {gender} | Region: {region}")
    
    results = {}
    
    # 2. 遍历款式文件夹 (e.g., "1.短袖T恤")
    for category_dir in os.listdir(root_path):
        cat_path = os.path.join(root_path, category_dir)
        if not os.path.isdir(cat_path): continue
        
        # 3. 遍历尺码文件夹 (e.g., "L", "M", "S")
        for size_dir in os.listdir(cat_path):
            size_path = os.path.join(cat_path, size_dir)
            if not os.path.isdir(size_path): continue
            
            # 查找该尺码下的图片
            image_candidates = [f for f in os.listdir(size_path) if f.lower().endswith(('.png', '.jpg'))]
            if not image_candidates: continue
            
            # 默认取第一张图片
            target_image = os.path.join(size_path, image_candidates[0])
            
            # 获取对应的人体数据
            body_specs = get_body_specs(gender, region, size_dir)
            
            # 唯一 ID
            uid = f"{category_dir}_{size_dir}"
            
            # 执行分析
            data = analyze_model_image(
                target_image, 
                category_name=category_dir, 
                size=size_dir, 
                specs=body_specs, 
                gender=gender, 
                region=region
            )
            
            if data:
                # 补充一些元数据
                data["original_file"] = target_image
                results[uid] = data
                
                # 实时保存，防止中断丢失
                save_path = os.path.join(cat_path, "size_descriptions.json")
                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            time.sleep(2) # 避免速率限制

# ================= 入口 =================
if __name__ == "__main__":
    # 替换为你实际的文件夹路径
    # 例如你截图里的那个 "female_eur_front_garment_with_model"
    target_folder = "path/to/female_eur_front_garment_with_model" 
    
    if os.path.exists(target_folder):
        process_root_folder(target_folder)
    else:
        print("Folder not found.")