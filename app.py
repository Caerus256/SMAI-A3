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
import streamlit as st
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


# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df, st.session_state.nutrition, st.session_state.idx_to_class, st.session_state.dish_to_region = load_metadata()
    st.session_state.current_bundle = None
    st.session_state.current_device = None
    st.session_state.current_model_type = None
    st.session_state.results = None

def get_available_models():
    return [m for m in MODEL_REGISTRY if m["available"]()]

def load_selected_model(model_type):
    bundle, device = get_model(model_type)
    if bundle is None:
        return False
    st.session_state.current_bundle = bundle
    st.session_state.current_device = device
    st.session_state.current_model_type = model_type
    return True

def classify_image(image, model_type):
    if image is None:
        return None

    # Load model if needed
    if st.session_state.current_model_type != model_type or st.session_state.current_bundle is None:
        ok = load_selected_model(model_type)
        if not ok:
            return None

    # Predict
    results = predict(image, st.session_state.current_bundle, st.session_state.current_device,
                     st.session_state.idx_to_class, st.session_state.dish_to_region,
                     st.session_state.nutrition, st.session_state.df)
    return results


# ─── Streamlit App ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Regional Thali Identifier",
        page_icon="🍛",
        layout="wide",
    )

    # ── Header ──
    st.title("🍛 Regional Thali Identifier")
    st.markdown("Upload a photo of an Indian dish — get the name, region, nutrition, and allergen info.")

    # ── Sidebar: model picker + stats ──
    available_models = get_available_models()
    if not available_models:
        st.error("No models available. Please train models first.")
        st.stop()

    model_choices = {m['label']: m['type'] for m in available_models}

    with st.sidebar:
        st.header("⚙️ Model Settings")
        selected_label = st.radio(
            "Choose a model for inference",
            options=list(model_choices.keys()),
            index=0,
        )
        selected_type = model_choices[selected_label]
        
        # Show detail caption for the selected model
        selected_info = next(m for m in available_models if m['type'] == selected_type)
        st.caption(f"**Architecture:** {selected_info['detail']}")

        st.divider()
        
        st.subheader("📊 Accuracy Reference")
        # Clean dataframe presentation of metrics to prevent truncation
        acc_df = pd.DataFrame([
            {
                "Model": m["label"], 
                "Top-1": f"{m['dish_acc']}%", 
                "Top-3": f"{m['top3_acc']}%", 
                "Region": f"{m['region_acc']}%"
            }
            for m in available_models
        ])
        st.dataframe(acc_df, hide_index=True, use_container_width=True)

    # ── Main area ──
    upload_col, result_col = st.columns([1, 2], gap="large")

    with upload_col:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Drop a food image here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, style="border-radius: 10px;")
            analyze_btn = st.button("✨ Analyze Dish", type="primary", use_container_width=True)
        else:
            image = None
            analyze_btn = False
            st.info("Upload a photo of an Indian dish to get started.")

    # ── Run inference ──
    if analyze_btn and image is not None:
        with st.spinner("Analyzing image features..."):
            st.session_state.results = classify_image(image, selected_type)

    # ── Display results ──
    with result_col:
        results = st.session_state.get("results")
        if results is None:
            st.subheader("🔍 Results")
            st.write("Results will appear here after analysis.")
            return

        if not results:
            st.error("Model weights not found. Train models first or check your internet connection for auto-download.")
            return

        top = results[0]

        # ── Dish name + confidence ──
        st.header(f"{top['name']}")
        
        # ── Region & Origin ──
        region_emoji = REGION_EMOJI.get(top["region"], "📍")
        st.markdown(f"**Origin:** {region_emoji} {top['region']} India &nbsp;·&nbsp; **State:** {top['state']}")
        
        st.progress(top["confidence"], text=f"Confidence: {top['confidence'] * 100:.1f}%")

        st.divider()

        # ── Key metrics row ──
        m1, m2, m3 = st.columns(3)
        m1.metric("Calories", f"~{top['calories']} kcal", help=f"Per {top.get('serving_size', 'serving')}")
        
        prep = top["prep_time"]
        cook = top["cook_time"]
        m2.metric("Prep Time", f"{prep} min" if prep and prep != "-1" else "—")
        m3.metric("Cook Time", f"{cook} min" if cook and cook != "-1" else "—")

        # ── Characteristics ──
        flavor = top["flavor"].title() if top["flavor"] and top["flavor"] != "-1" else "—"
        st.markdown(
            f"**Diet:** {top['diet'].title()} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Course:** {top['course'].title()} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Flavor:** {flavor}"
        )

        st.divider()

        # ── Allergens ──
        st.markdown("#### 🚨 Allergens")
        a1, a2, a3 = st.columns(3)
        allergens = top.get("allergens", {})
        
        if allergens.get("nuts", False):
            a1.error("🥜 Nuts: Contains")
        else:
            a1.success("🥜 Nuts: Free")
            
        if allergens.get("dairy", False):
            a2.error("🥛 Dairy: Contains")
        else:
            a2.success("🥛 Dairy: Free")
            
        if allergens.get("gluten", False):
            a3.error("🌾 Gluten: Contains")
        else:
            a3.success("🌾 Gluten: Free")

        st.write("") # Small spacer

        # ── Ingredients ──
        with st.expander("📝 View Ingredients", expanded=False):
            st.write(top["ingredients"])
            st.caption(f"Serving size: {top.get('serving_size', '1 serving')}")

        # ── Alternative predictions ──
        if len(results) > 1:
            st.divider()
            st.markdown("#### 🔮 Alternative Predictions")
            for rank, r in enumerate(results[1:], 2):
                alt_col1, alt_col2, alt_col3 = st.columns([3, 2, 1])
                alt_col1.write(f"**#{rank}** {r['name']}")
                alt_col2.write(f"{REGION_EMOJI.get(r['region'], '📍')} {r['region']} India")
                alt_col3.write(f"{r['confidence'] * 100:.1f}%")

if __name__ == "__main__":
    main()

