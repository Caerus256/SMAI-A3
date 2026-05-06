# Regional Thali Identifier - Streamlit App

import streamlit as st
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
import time
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
    "North":      "#b37a50",
    "South":      "#5c9968",
    "East":       "#a89050",
    "West":       "#607ea0",
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
        "dish_acc": 74.8, "top3_acc": 89.7, "region_acc": 84.0,
        "available": lambda: _OPEN_CLIP_AVAILABLE,
    },
    {
        "type": "vit",   "rank": 2,
        "label": "ViT-B/16 Fine-tuned",
        "detail": "Last-2-blocks, 10 epochs",
        "dish_acc": 69.5, "top3_acc": 85.8, "region_acc": 82.2,
        "available": lambda: True,
    },
    {
        "type": "efficientnet", "rank": 3,
        "label": "EfficientNet-B0",
        "detail": "Last-block, 10 epochs",
        "dish_acc": 49.0, "top3_acc": 70.3, "region_acc": 69.3,
        "available": lambda: True,
    },
]

RANK_MEDAL = {1: "1st", 2: "2nd", 3: "3rd"}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Page config
st.set_page_config(
    page_title="Regional Thali Identifier",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS — Enhanced muted dark palette with animations
CUSTOM_CSS = """
<style>
:root {
    --bg:     #0b0b0d;
    --card:   #161618;
    --border: #2a2a30;
    --text1:  #e0e0e0;
    --text2:  #9898a0;
    --text3:  #7a7a85; /* Brightened for better accessibility */
    --accent: #d4aa78; /* Slightly brighter accent */
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in-section {
    animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.stApp { background-color: var(--bg) !important; font-family: 'Inter', -apple-system, sans-serif !important; }

section[data-testid="stSidebar"] {
    background-color: #121214 !important;
    border-right: 1px solid var(--border) !important;
}

h1,h2,h3,h4,h5,h6 { color: var(--text1) !important; }
p,span,div,label,li { color: var(--text1) !important; }

.section-label {
    color: var(--text3) !important;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 1.5rem 0 0.8rem;
}

.app-header {
    text-align: center;
    padding: 1.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text1) !important;
    margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
}
.app-title-accent { color: var(--accent) !important; }
.app-sub {
    color: var(--text2) !important;
    font-size: 0.95rem;
    margin: 0;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}

.model-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin: 0.3rem 0;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}
.model-row:hover { background: #1c1c20; border-color: var(--border); }
.model-row.active {
    background: #1e1b15;
    border-color: #4a4030;
}
.model-rank {
    font-size: 0.7rem;
    font-weight: 800;
    color: var(--text3) !important;
    min-width: 28px;
    text-align: center;
    padding: 0.2rem 0;
    border-radius: 6px;
    background: #232328;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.model-row.active .model-rank {
    color: var(--accent) !important;
    background: #2e261a;
}
.model-name { font-size: 0.9rem; font-weight: 600; color: var(--text1) !important; }
.model-detail { font-size: 0.75rem; color: var(--text3) !important; margin-top: 2px; }
.model-acc { font-size: 0.85rem; font-weight: 700; color: var(--text2) !important; min-width: 50px; text-align: right; }
.model-row.active .model-acc { color: var(--accent) !important; }
.model-bar-bg { height: 4px; background: #232328; border-radius: 2px; margin-top: 0.5rem; }
.model-bar { height: 4px; border-radius: 2px; background: var(--text3); transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1); }
.model-row.active .model-bar { background: var(--accent); }
.model-unavail { opacity: 0.4; pointer-events: none; filter: grayscale(100%); }

.region-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    margin: 0.5rem 0;
}

.conf-bg { background: #232328; border-radius: 4px; height: 8px; overflow: hidden; }
.conf-fill { height: 8px; border-radius: 4px; background: #a6937c; transition: width 1s ease-out; }

.cal-card {
    background: #1c1a18;
    border: 1px solid #363028;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.5rem 0;
}
.cal-num { font-size: 2.2rem; font-weight: 800; color: var(--accent) !important; line-height: 1; }
.cal-sub { color: var(--text2) !important; font-size: 0.8rem; margin-top: 0.4rem; font-weight: 500; }

.allergen-badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
    margin: 0.2rem 0.2rem;
}
.allergen-yes { background:#2a1a1a; color:#e08a8a !important; border:1px solid #4a2a2a; }
.allergen-no  { background:#1a2a1a; color:#8ae08a !important; border:1px solid #2a4a2a; }

.pill {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
    background: #1f1f24;
    color: var(--text1) !important;
    border: 1px solid var(--border);
}
.pill-veg  { background:#1a281a; color:#8cd98c !important; border-color:#2d4d2d; }
.pill-nveg { background:#281a1a; color:#d98c8c !important; border-color:#4d2d2d; }

.welcome {
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 3rem auto;
    max-width: 560px;
    transition: border-color 0.3s ease;
}
.welcome:hover { border-color: var(--text3); }
.welcome-icon { font-size: 4rem; margin-bottom: 0.5rem; }

div[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed #3a3a45 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
div[data-testid="stFileUploader"] * { color: var(--text2) !important; }

/* Clean up native buttons used for selection */
.model-btn > button, .sample-btn > button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text1) !important;
    transition: all 0.2s ease !important;
}
.model-btn > button:hover, .sample-btn > button:hover {
    border-color: #4a4a55 !important;
    background: #1f1f24 !important;
}
.model-btn-active > button {
    background: #1e1b15 !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    font-weight: bold !important;
}
.sample-btn-active > button {
    background: var(--accent) !important;
    color: #000 !important;
    border-color: var(--accent) !important;
    font-weight: 700 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; max-width: 1200px; }
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.streamlit-expanderHeader { color: var(--text1) !important; font-weight: 600 !important; font-size: 0.95rem !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Download model from Hugging Face if not present
def ensure_model(filename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / filename
    if not local_path.exists():
        try:
            st.toast(f"Downloading {filename} from Hugging Face...", icon="⬇️")
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="model",
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            st.toast(f"Downloaded {filename}", icon="✅")
        except Exception as e:
            st.error(f"Failed to download {filename} from Hugging Face: {str(e)}")
            st.info("Models are not available. Please run training.ipynb locally to generate model files.")
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
@st.cache_data
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


# Sidebar
def render_sidebar(df):
    st.sidebar.markdown(
        '<div style="text-align:center; padding:1.2rem 0 0.8rem;">'
        '<span style="font-size:2rem;">🍛</span>'
        '<p style="font-size:1.1rem; font-weight:800; color:#e0e0e0 !important;'
        ' margin:0.3rem 0 0; letter-spacing:0.02em;">Thali Identifier</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<hr style="border:0; border-top:1px solid var(--border); margin:0 0 1rem 0;">', unsafe_allow_html=True)

    st.sidebar.markdown('<p class="section-label">Model Engine</p>', unsafe_allow_html=True)

    available = [m for m in MODEL_REGISTRY if m["available"]()]
    if not available:
        st.sidebar.error("No models found. Run the training notebook first.")
        return None

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = available[0]["type"]

    for m in MODEL_REGISTRY:
        avail = m["available"]()
        is_active = avail and m["type"] == st.session_state["selected_model"]
        rank_str = RANK_MEDAL[m["rank"]]

        row_cls = "model-row active" if is_active else "model-row"
        if not avail:
            row_cls += " model-unavail"
        st.sidebar.markdown(
            f'<div class="{row_cls}">'
            f'  <div><span class="model-rank">{rank_str}</span></div>'
            f'  <div style="flex:1; min-width:0;">'
            f'    <div class="model-name">{m["label"]}</div>'
            f'    <div class="model-detail">{m["detail"]}</div>'
            f'    <div class="model-bar-bg">'
            f'      <div class="model-bar" style="width:{m["dish_acc"]}%;"></div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="model-acc">{m["dish_acc"]}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if avail:
            btn_cls = "model-btn-active" if is_active else "model-btn"
            btn_label = "Active Engine" if is_active else f"Select {m['label']}"
            with st.sidebar:
                st.markdown(f'<div class="{btn_cls}">', unsafe_allow_html=True)
                clicked = st.button(
                    btn_label, key=f"sel_{m['type']}",
                    use_container_width=True,
                    disabled=is_active,
                )
                st.markdown('</div>', unsafe_allow_html=True)
                if clicked:
                    st.session_state["selected_model"] = m["type"]
                    st.rerun()

    chosen = next(m for m in available if m["type"] == st.session_state["selected_model"])

    st.sidebar.markdown(
        '<div style="display:flex; justify-content:space-around; padding:1rem 0;'
        ' border-top:1px solid var(--border); border-bottom:1px solid var(--border); margin-top:1rem;">'
        '<div style="text-align:center;">'
        '  <div style="font-size:0.65rem; color:var(--text3); text-transform:uppercase; font-weight:700;">Top-1</div>'
        f'  <div style="font-size:1rem; font-weight:800; color:var(--accent);">{chosen["dish_acc"]}%</div>'
        '</div>'
        '<div style="text-align:center;">'
        '  <div style="font-size:0.65rem; color:var(--text3); text-transform:uppercase; font-weight:700;">Top-3</div>'
        f'  <div style="font-size:1rem; font-weight:800; color:var(--accent);">{chosen["top3_acc"]}%</div>'
        '</div>'
        '<div style="text-align:center;">'
        '  <div style="font-size:0.65rem; color:var(--text3); text-transform:uppercase; font-weight:700;">Region</div>'
        f'  <div style="font-size:1rem; font-weight:800; color:var(--accent);">{chosen["region_acc"]}%</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<p class="section-label" style="margin-top:2rem;">Try a sample</p>', unsafe_allow_html=True)
    sample_dishes = ["biryani", "butter_chicken", "jalebi", "palak_paneer", "daal_baati_churma", "modak"]
    
    cols = st.sidebar.columns(2) # Switched to 2 columns for better button size
    for i, dish in enumerate(sample_dishes):
        dish_dir = IMG_DIR / dish
        if dish_dir.exists():
            imgs = sorted(os.listdir(dish_dir))
            if imgs:
                dn = df[df["folder_name"] == dish]["display_name"].values
                name = dn[0] if len(dn) else dish
                
                # Active state visual check
                is_active_sample = st.session_state.get("sample_dish_name") == dish
                btn_class = "sample-btn-active" if is_active_sample else "sample-btn"
                
                with cols[i % 2]:
                    st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
                    if st.button(name[:12], key=f"s_{dish}", use_container_width=True):
                        st.session_state["sample_image"] = str(dish_dir / imgs[0])
                        st.session_state["sample_dish_name"] = dish
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    return chosen["type"]


# Main
def main():
    df, nutrition, idx_to_class, dish_to_region = load_metadata()

    model_type = render_sidebar(df)
    if model_type is None:
        st.stop()

    bundle, device = get_model(model_type)
    if bundle is None:
        st.error("Model weights not found.")
        st.stop()

    # Header
    st.markdown(
        '<div class="app-header fade-in-section">'
        '  <h1 class="app-title">'
        '    <span class="app-title-accent">Regional</span> Thali Identifier'
        '  </h1>'
        '  <p class="app-sub">'
        '    Upload an Indian food photo &mdash; instantly uncover the dish, origin, nutrition, and allergens.'
        '  </p>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop a food photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    
    image = None
    if uploaded:
        image = Image.open(uploaded)
        # Clear sample state if user uploads their own
        if "sample_dish_name" in st.session_state:
            del st.session_state["sample_dish_name"]
    elif "sample_image" in st.session_state:
        image = Image.open(st.session_state["sample_image"])

    if image is None:
        st.markdown(
            '<div class="welcome fade-in-section">'
            '  <div class="welcome-icon">📸</div>'
            '  <h3 style="margin:0.5rem 0 0.8rem; font-weight:700; font-size:1.3rem;">'
            '    Drop a photo above or pick a sample from the menu</h3>'
            '  <p style="color:var(--text2) !important; font-size:0.95rem; margin:0;">'
            '    High-quality JPG and PNG supported.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Inference Wrapper
    with st.spinner("Analyzing spices, textures, and regions..."):
        results = predict(image, bundle, device, idx_to_class, dish_to_region, nutrition, df)
        # Add slight sleep if it's too fast to allow user to register the loading state
        time.sleep(0.3) 
        
    st.toast('Analysis Complete!', icon='✅')

    top = results[0]

    st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.image(image, use_container_width=True, output_format="JPEG")
        
        # Move ingredients to left column under the image for better balance
        with st.expander("📝 View Full Ingredients", expanded=False):
            st.markdown(
                f'<p style="font-size:0.9rem; color:var(--text1) !important;'
                f' line-height:1.8;">{top["ingredients"]}</p>',
                unsafe_allow_html=True,
            )

    with col_right:
        rc = REGION_COLORS.get(top["region"], "#78787e")
        re_emoji = REGION_EMOJI.get(top["region"], "🍽️")

        conf_pct = top["confidence"] * 100
        st.markdown(
            '<div class="card">'
            '  <div style="font-size:0.7rem; color:var(--text3); text-transform:uppercase;'
            '   letter-spacing:0.12em; font-weight:700; margin-bottom:0.5rem;">AI Prediction</div>'
            f'  <div style="font-size:2rem; font-weight:800; color:var(--text1);'
            f'   line-height:1.2; margin-bottom:0.8rem;">{top["name"]}</div>'
            f'  <span class="region-badge"'
            f'   style="background:{rc}15; color:{rc}; border:1px solid {rc}40;">'
            f'    {re_emoji} {top["region"]} India &middot; {top["state"]}'
            f'  </span>'
            f'  <div style="margin-top:1.2rem;">'
            f'    <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">'
            f'      <span style="font-size:0.85rem; color:var(--text2); font-weight:500;">Confidence Score</span>'
            f'      <span style="font-size:0.85rem; font-weight:700; color:var(--accent);">{conf_pct:.1f}%</span>'
            f'    </div>'
            f'    <div class="conf-bg">'
            f'      <div class="conf-fill" style="width:{conf_pct:.1f}%;"></div>'
            f'    </div>'
            f'  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            raw_cal = top["calories"]
            try:
                cal_disp = f"{int(raw_cal):,}"
            except (ValueError, TypeError):
                cal_disp = str(raw_cal)

            st.markdown(
                '<div class="cal-card">'
                f'  <div class="cal-num">~{cal_disp}</div>'
                f'  <div class="cal-sub">kcal per {top["serving_size"]}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            diet = top["diet"].lower()
            dcls = "pill-veg" if "veg" in diet and "non" not in diet else "pill-nveg"
            st.markdown(
                '<div style="margin-top:0.8rem; display:flex; flex-wrap:wrap; gap:4px;">'
                f'  <span class="pill {dcls}">{top["diet"].title()}</span>'
                f'  <span class="pill">{top["course"].title()}</span>'
                f'  <span class="pill">{top["flavor"].title()}</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                '<div class="card" style="padding:1.2rem;">'
                '<div style="font-size:0.7rem; color:var(--text3); text-transform:uppercase;'
                ' font-weight:700; letter-spacing:0.1em; margin-bottom:0.8rem;">Allergens</div>',
                unsafe_allow_html=True,
            )
            amap = {"nuts": "🥜 Nuts", "dairy": "🥛 Dairy", "gluten": "🌾 Gluten"}
            allergens = top.get("allergens", {})
            for key, label in amap.items():
                present = allergens.get(key, False)
                cls = "allergen-yes" if present else "allergen-no"
                txt = "Contains" if present else "Free"
                st.markdown(
                    f'<div style="margin-bottom:6px;"><span class="allergen-badge {cls}" style="width:100%; text-align:center;">{label} &middot; {txt}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div style="margin-top:0.8rem; font-size:0.85rem; color:var(--text2); font-weight:500; text-align:center;">'
                f'  ⏱️ Prep: {top["prep_time"]}m &nbsp;&middot;&nbsp; Cook: {top["cook_time"]}m'
                f'</div>',
                unsafe_allow_html=True,
            )

        if len(results) > 1:
            st.markdown('<p class="section-label" style="margin-top:2rem;">Alternative Predictions</p>', unsafe_allow_html=True)
            for rank, r in enumerate(results[1:], 2):
                pct = r["confidence"] * 100
                st.markdown(
                    '<div class="card" style="padding:0.8rem 1.2rem; display:flex; align-items:center; gap:1rem; margin-bottom:0.4rem;">'
                    f'  <span style="font-size:0.85rem; font-weight:800; color:var(--text3);'
                    f'   min-width:24px; text-align:center;">#{rank}</span>'
                    f'  <div style="flex:1;">'
                    f'    <span style="font-size:0.95rem; font-weight:700; color:var(--text1);">'
                    f'      {r["name"]}</span>'
                    f'    <span style="font-size:0.8rem; color:var(--text3); font-weight:500; margin-left:0.6rem;">'
                    f'      {r["region"]}</span>'
                    f'  </div>'
                    f'  <span style="font-size:0.9rem; font-weight:700; color:var(--text2);">'
                    f'    {pct:.1f}%</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                
    st.markdown('</div>', unsafe_allow_html=True) # End fade-in section

if __name__ == "__main__":
    main()