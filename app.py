# Regional Thali Identifier - Gradio App

import torch
import timm
import json
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
import gradio as gr
from huggingface_hub import hf_hub_download

try:
    import open_clip
    _OPEN_CLIP_AVAILABLE = True
except ImportError:
    _OPEN_CLIP_AVAILABLE = False

# Paths & constants
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
MODEL_DIR   = PROJECT_DIR / "model"
IMG_DIR     = Path(os.environ.get("IMG_DIR", str(PROJECT_DIR / "Indian Food Images")))
HF_REPO_ID  = "Caerus256/SMAI-A3"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 80

REGION_COLORS = {
    "North": "#b37a50",
    "South": "#5c9968",
    "East": "#a89050",
    "West": "#607ea0",
    "North East": "#887aaa",
}
REGION_EMOJI = {
    "North": "🏔️", "South": "🌴", "East": "🌊",
    "West": "🏜️",  "North East": "🍃",
}

MODEL_REGISTRY = [
    {
        "type": "clip",  "rank": 1,
        "label": "CLIP Linear Probe",
        "detail": "ViT-B/32 + Logistic Regression",
        "dish_acc": 74.83, "top3_acc": 89.67, "region_acc": 84.00,
        "available": lambda: _OPEN_CLIP_AVAILABLE,
    },
    {
        "type": "vit",   "rank": 2,
        "label": "ViT-B/16 Fine-tuned",
        "detail": "Last-2-blocks, 10 epochs",
        "dish_acc": 69.33, "top3_acc": 85.17, "region_acc": 82.50,
        "available": lambda: True,
    },
    {
        "type": "efficientnet", "rank": 3,
        "label": "EfficientNet-B0",
        "detail": "Last-block, 10 epochs",
        "dish_acc": 49.83, "top3_acc": 72.33, "region_acc": 69.50,
        "available": lambda: True,
    },
]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Download model from Hugging Face if not present
def ensure_model(filename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / filename
    if not local_path.exists():
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="model",
                local_dir=MODEL_DIR
            )
        except Exception as e:
            return None
    return local_path


# Model loaders
def _load_clip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = ensure_model("clip_linear_probe.pkl")
    if p is None:
        return None, device
    with open(p, "rb") as fh:
        clf = pickle.load(fh)
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device).eval()
    return {"type": "clip", "clf": clf, "clip_model": clip_model, "clip_prep": clip_prep}, device


def _load_vit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = ensure_model("vit_b16_best.pth")
    if p is None:
        return None, device
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    sd = torch.load(str(p), map_location=device, weights_only=True)
    model.load_state_dict(sd)
    return {"type": "vit", "model": model.to(device).eval()}, device


def _load_efficientnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ckpt in ["efficientnet_b0_best.pth", "efficientnet_b0_head_only.pth"]:
        p = ensure_model(ckpt)
        if p and p.exists():
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
            sd = torch.load(str(p), map_location=device, weights_only=True)
            model.load_state_dict(sd)
            return {"type": "efficientnet", "model": model.to(device).eval()}, device
    return None, device


def get_model(model_type):
    if model_type == "clip":
        return _load_clip()
    if model_type == "vit":
        return _load_vit()
    return _load_efficientnet()


# Metadata
def load_metadata():
    df = pd.read_csv(DATA_DIR / "food_metadata_80.csv")
    with open(DATA_DIR / "nutrition_allergens.json", "r", encoding="utf-8") as fh:
        nutrition = json.load(fh)
    class_names = sorted(df["folder_name"].tolist())
    idx_to_class = {i: n for i, n in enumerate(class_names)}
    dish_to_region = dict(zip(df["folder_name"], df["region"]))
    return df, nutrition, idx_to_class, dish_to_region


# Prediction
def _build_result(idx, prob, idx_to_class, dish_to_region, nutrition, df):
    folder = idx_to_class[int(idx)]
    info = nutrition.get(folder, {})
    row = df[df["folder_name"] == folder].iloc[0]
    return {
        "folder":       folder,
        "name":         info.get("display_name",     row["display_name"]),
        "confidence":   float(prob),
        "region":       dish_to_region.get(folder, "Unknown"),
        "state":        info.get("state",            row.get("state", "")),
        "diet":         info.get("diet",             row.get("diet", "")),
        "course":       info.get("course",           row.get("course", "")),
        "flavor":       info.get("flavor_profile",   row.get("flavor_profile", "")),
        "ingredients":  info.get("ingredients",      row.get("ingredients", "")),
        "prep_time":    info.get("prep_time_min",    str(row.get("prep_time", ""))),
        "cook_time":    info.get("cook_time_min",    str(row.get("cook_time", ""))),
        "allergens":    info.get("allergens",        {}),
        "calories":     info.get("approx_calories_kcal", "N/A"),
        "serving_size": info.get("serving_size",     "1 serving"),
    }


def predict(image, bundle, device, idx_to_class, dish_to_region, nutrition, df, top_k=3):
    rgb = image.convert("RGB")
    if bundle["type"] == "clip":
        img_t = bundle["clip_prep"](rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = bundle["clip_model"].encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        probs = bundle["clf"].predict_proba(feat.cpu().numpy())[0]
        top_i = probs.argsort()[::-1][:top_k]
        return [_build_result(i, probs[i], idx_to_class, dish_to_region, nutrition, df)
                for i in top_i]
    else:
        img_t = VAL_TRANSFORM(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = bundle["model"](img_t)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        top_p, top_i = probs.topk(top_k)
        return [_build_result(i.item(), p.item(), idx_to_class, dish_to_region, nutrition, df)
                for p, i in zip(top_p, top_i)]


# Global state
df, nutrition, idx_to_class, dish_to_region = load_metadata()
current_bundle = None
current_device = None
current_model_type = None

def get_available_models():
    return [m for m in MODEL_REGISTRY if m["available"]()]

def load_selected_model(model_type):
    global current_bundle, current_device, current_model_type
    bundle, device = get_model(model_type)
    if bundle is None:
        return None, "❌ Model weights not found. Please train models first."
    current_bundle = bundle
    current_device = device
    current_model_type = model_type
    model_info = next(m for m in MODEL_REGISTRY if m["type"] == model_type)
    return f"✅ Loaded: {model_info['label']} ({model_info['detail']})"

def classify_image(image, model_type):
    global current_bundle, current_device, current_model_type
    
    if image is None:
        return None, "Please upload an image first.", ""
    
    # Load model if needed
    if current_model_type != model_type or current_bundle is None:
        load_msg = load_selected_model(model_type)
        if "❌" in load_msg:
            return None, load_msg, ""
    
    # Predict
    results = predict(image, current_bundle, current_device, idx_to_class, dish_to_region, nutrition, df)
    
    # Format output
    top = results[0]
    
    # Main prediction
    main_output = f"""
    ## 🍛 {top['name']}
    
    **Region:** {REGION_EMOJI.get(top['region'], '🍽️')} {top['region']} India · {top['state']}
    
    **Confidence:** {top['confidence'] * 100:.1f}%
    
    ### 📊 Nutrition
    **Calories:** ~{top['calories']} kcal per {top['serving_size']}
    
    ### 🥗 Diet & Course
    **Diet:** {top['diet'].title()}  
    **Course:** {top['course'].title()}  
    **Flavor:** {top['flavor'].title()}
    
    ### ⏱️ Time
    **Prep:** {top['prep_time']}m · **Cook:** {top['cook_time']}m
    
    ### 🚨 Allergens
    """
    
    allergens = top.get('allergens', {})
    allergen_info = []
    if allergens.get('nuts', False):
        allergen_info.append("🥜 **Nuts:** Contains")
    else:
        allergen_info.append("🥜 **Nuts:** Free")
    
    if allergens.get('dairy', False):
        allergen_info.append("🥛 **Dairy:** Contains")
    else:
        allergen_info.append("🥛 **Dairy:** Free")
    
    if allergens.get('gluten', False):
        allergen_info.append("🌾 **Gluten:** Contains")
    else:
        allergen_info.append("🌾 **Gluten:** Free")
    
    main_output += "\n".join(allergen_info)
    
    # Alternative predictions
    if len(results) > 1:
        main_output += "\n\n### 🔮 Alternative Predictions\n"
        for rank, r in enumerate(results[1:], 2):
            main_output += f"**#{rank}** {r['name']} ({r['region']}) - {r['confidence'] * 100:.1f}%\n"
    
    # Ingredients
    ingredients_output = f"### 📝 Ingredients\n\n{top['ingredients']}"
    
    return image, main_output, ingredients_output

# Create Gradio interface
def create_interface():
    available_models = get_available_models()
    if not available_models:
        raise ValueError("No models available. Please train models first.")
    
    model_choices = [f"{m['rank']}. {m['label']}" for m in available_models]
    model_type_map = {f"{m['rank']}. {m['label']}": m['type'] for m in available_models}
    
    with gr.Blocks(title="Regional Thali Identifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🍛 Regional Thali Identifier
        Upload an Indian food photo &mdash; instantly uncover the dish, origin, nutrition, and allergens.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Food Photo")
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Select Model",
                    info="Choose the AI model for prediction"
                )
                classify_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                gr.Markdown("### Model Performance")
                for m in available_models:
                    gr.Markdown(f"**{m['label']}** - Top-1: {m['dish_acc']}%, Top-3: {m['top3_acc']}%, Region: {m['region_acc']}%")
            
            with gr.Column(scale=2):
                output_image = gr.Image(label="Uploaded Image")
                prediction_output = gr.Markdown(label="Prediction Results")
                ingredients_output = gr.Markdown(label="Ingredients")
        
        classify_btn.click(
            fn=lambda img, model: classify_image(img, model_type_map[model]),
            inputs=[image_input, model_dropdown],
            outputs=[output_image, prediction_output, ingredients_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                [str(IMG_DIR / "biryani" / os.listdir(IMG_DIR / "biryani")[0]) if (IMG_DIR / "biryani").exists() else None],
                [str(IMG_DIR / "butter_chicken" / os.listdir(IMG_DIR / "butter_chicken")[0]) if (IMG_DIR / "butter_chicken").exists() else None],
                [str(IMG_DIR / "jalebi" / os.listdir(IMG_DIR / "jalebi")[0]) if (IMG_DIR / "jalebi").exists() else None],
            ],
            inputs=image_input,
            label="Sample Images"
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
