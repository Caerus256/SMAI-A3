# Regional Thali Identifier — Streamlit App

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

try:
    import open_clip
    _OPEN_CLIP_AVAILABLE = True
except ImportError:
    _OPEN_CLIP_AVAILABLE = False

# -- Paths & constants --------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
MODEL_DIR   = PROJECT_DIR / "model"
IMG_DIR     = Path(os.environ.get("IMG_DIR", str(PROJECT_DIR.parent / "Indian Food Images")))

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
        "available": lambda: _OPEN_CLIP_AVAILABLE and (MODEL_DIR / "clip_linear_probe.pkl").exists(),
    },
    {
        "type": "vit",   "rank": 2,
        "label": "ViT-B/16 Fine-tuned",
        "detail": "Last-2-blocks, 10 epochs",
        "dish_acc": 69.5, "top3_acc": 85.8, "region_acc": 82.2,
        "available": lambda: (MODEL_DIR / "vit_b16_best.pth").exists(),
    },
    {
        "type": "efficientnet", "rank": 3,
        "label": "EfficientNet-B0",
        "detail": "Last-block, 10 epochs",
        "dish_acc": 49.0, "top3_acc": 70.3, "region_acc": 69.3,
        "available": lambda: any(
            (MODEL_DIR / f).exists()
            for f in ["efficientnet_b0_best.pth", "efficientnet_b0_head_only.pth"]
        ),
    },
]

RANK_MEDAL = {1: "1st", 2: "2nd", 3: "3rd"}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Regional Thali Identifier",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS — muted dark palette --------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg:     #0f0f0f;
    --card:   #161618;
    --border: #232328;
    --text1:  #d0d0d0;
    --text2:  #78787e;
    --text3:  #505058;
    --accent: #b8956a;
}

.stApp { background-color: var(--bg) !important; }

section[data-testid="stSidebar"] {
    background-color: #121214 !important;
    border-right: 1px solid var(--border) !important;
}

h1,h2,h3,h4,h5,h6 { color: var(--text1) !important; }
p,span,div,label,li { color: var(--text1) !important; }

.section-label {
    color: var(--text3) !important;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.4rem 0 0.5rem;
}

.app-header {
    text-align: center;
    padding: 0.8rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.app-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text1) !important;
    margin: 0 0 0.25rem;
    letter-spacing: -0.01em;
}
.app-title-accent { color: var(--accent) !important; }
.app-sub {
    color: var(--text2) !important;
    font-size: 0.82rem;
    margin: 0;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
}

.model-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.65rem 0.8rem;
    border-radius: 8px;
    margin: 0.2rem 0;
    border: 1px solid transparent;
    transition: all 0.15s ease;
}
.model-row:hover { background: #1c1c20; }
.model-row.active {
    background: #1a1914;
    border-color: #3a352a;
}
.model-rank {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text3) !important;
    min-width: 22px;
    text-align: center;
    padding: 0.15rem 0;
    border-radius: 4px;
    background: #1e1e22;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.model-row.active .model-rank {
    color: var(--accent) !important;
    background: #252018;
}
.model-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text1) !important;
}
.model-detail {
    font-size: 0.68rem;
    color: var(--text3) !important;
    margin-top: 1px;
}
.model-acc {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text2) !important;
    min-width: 46px;
    text-align: right;
}
.model-row.active .model-acc { color: var(--accent) !important; }
.model-bar-bg {
    height: 3px;
    background: #1e1e22;
    border-radius: 2px;
    margin-top: 0.35rem;
}
.model-bar {
    height: 3px;
    border-radius: 2px;
    background: var(--text3);
    transition: width 0.3s ease;
}
.model-row.active .model-bar { background: var(--accent); }
.model-unavail { opacity: 0.3; pointer-events: none; }

.region-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
    margin: 0.25rem 0;
}

.conf-bg {
    background: #1e1e22;
    border-radius: 4px;
    height: 6px;
}
.conf-fill {
    height: 6px;
    border-radius: 4px;
    background: #8a7a6a;
}

.cal-card {
    background: #1a1816;
    border: 1px solid #2a2620;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin: 0.4rem 0;
}
.cal-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent) !important;
    line-height: 1.1;
}
.cal-sub {
    color: var(--text3) !important;
    font-size: 0.72rem;
    margin-top: 0.15rem;
}

.allergen-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.75rem;
    margin: 0.15rem 0.15rem;
}
.allergen-yes { background:#261818; color:#c08080 !important; border:1px solid #3a2828; }
.allergen-no  { background:#182618; color:#80b080 !important; border:1px solid #283a28; }

.pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 5px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.12rem;
    background: #1a1a1e;
    color: var(--text2) !important;
    border: 1px solid var(--border);
}
.pill-veg  { background:#162016; color:#7aaa7a !important; border-color:#2a382a; }
.pill-nveg { background:#201616; color:#aa7a7a !important; border-color:#382a2a; }

.welcome {
    background: var(--card);
    border: 1px dashed var(--border);
    border-radius: 14px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 520px;
}
.welcome-icon { font-size: 3rem; margin-bottom: 0.4rem; }

div[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px dashed #303038 !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploader"] * { color: var(--text3) !important; }
.stRadio > div { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; max-width: 1200px; }
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
.streamlit-expanderHeader { color: var(--text2) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -- Cached model loaders -----------------------------------------------------
@st.cache_resource
def _load_clip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(MODEL_DIR / "clip_linear_probe.pkl", "rb") as fh:
        clf = pickle.load(fh)
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(device).eval()
    return {"type": "clip", "clf": clf, "clip_model": clip_model, "clip_prep": clip_prep}, device


@st.cache_resource
def _load_vit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    sd = torch.load(str(MODEL_DIR / "vit_b16_best.pth"), map_location=device, weights_only=True)
    model.load_state_dict(sd)
    return {"type": "vit", "model": model.to(device).eval()}, device


@st.cache_resource
def _load_efficientnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ckpt in ["efficientnet_b0_best.pth", "efficientnet_b0_head_only.pth"]:
        p = MODEL_DIR / ckpt
        if p.exists():
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


# -- Metadata ------------------------------------------------------------------
@st.cache_data
def load_metadata():
    df = pd.read_csv(DATA_DIR / "food_metadata_80.csv")
    with open(DATA_DIR / "nutrition_allergens.json", "r", encoding="utf-8") as fh:
        nutrition = json.load(fh)
    class_names = sorted(df["folder_name"].tolist())
    idx_to_class = {i: n for i, n in enumerate(class_names)}
    dish_to_region = dict(zip(df["folder_name"], df["region"]))
    return df, nutrition, idx_to_class, dish_to_region


# -- Prediction ----------------------------------------------------------------
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


# -- Sidebar -------------------------------------------------------------------
def _model_row_html(m, is_active):
    cls = "model-row active" if is_active else "model-row"
    if not m["available"]():
        cls += " model-unavail"
    rank_str = RANK_MEDAL[m["rank"]]
    return (
        f'<div class="{cls}">'
        f'  <div><span class="model-rank">{rank_str}</span></div>'
        f'  <div style="flex:1; min-width:0;">'
        f'    <div class="model-name">{m["label"]}</div>'
        f'    <div class="model-detail">{m["detail"]}</div>'
        f'    <div class="model-bar-bg"><div class="model-bar" style="width:{m["dish_acc"]}%;"></div></div>'
        f'  </div>'
        f'  <div class="model-acc">{m["dish_acc"]}%</div>'
        f'</div>'
    )


def render_sidebar(df):
    # Logo
    st.sidebar.markdown(
        '<div style="text-align:center; padding:0.8rem 0 0.4rem;">'
        '<span style="font-size:1.6rem;">🍛</span>'
        '<p style="font-size:0.9rem; font-weight:700; color:#d0d0d0 !important;'
        ' margin:0.15rem 0 0; letter-spacing:0.02em;">Thali Identifier</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div style="border-top:1px solid #232328; margin:0.3rem 0 0.8rem;"></div>',
        unsafe_allow_html=True,
    )

    # -- Unified model picker --------------------------------------------------
    st.sidebar.markdown(
        '<p class="section-label">Model</p>', unsafe_allow_html=True
    )

    available = [m for m in MODEL_REGISTRY if m["available"]()]
    if not available:
        st.sidebar.error("No models found. Run the training notebook first.")
        return None

    model_labels = [m["label"] for m in available]
    sel_idx = st.sidebar.radio(
        "model_sel", range(len(available)),
        format_func=lambda i: model_labels[i],
        label_visibility="collapsed",
    )
    chosen = available[sel_idx]

    # Visual rows for ALL models (available + unavailable)
    for m in MODEL_REGISTRY:
        is_active = m["available"]() and m["type"] == chosen["type"]
        st.sidebar.markdown(_model_row_html(m, is_active), unsafe_allow_html=True)

    # Stats row for selected model
    st.sidebar.markdown(
        '<div style="display:flex; justify-content:space-between; padding:0.5rem 0.8rem 0;'
        ' border-top:1px solid #232328; margin-top:0.5rem;">'
        '<div style="text-align:center;">'
        '  <div style="font-size:0.6rem; color:#505058; text-transform:uppercase;'
        '   letter-spacing:0.08em;">Top-1</div>'
        f'  <div style="font-size:0.85rem; font-weight:700; color:#b8956a;">{chosen["dish_acc"]}%</div>'
        '</div>'
        '<div style="text-align:center;">'
        '  <div style="font-size:0.6rem; color:#505058; text-transform:uppercase;'
        '   letter-spacing:0.08em;">Top-3</div>'
        f'  <div style="font-size:0.85rem; font-weight:700; color:#b8956a;">{chosen["top3_acc"]}%</div>'
        '</div>'
        '<div style="text-align:center;">'
        '  <div style="font-size:0.6rem; color:#505058; text-transform:uppercase;'
        '   letter-spacing:0.08em;">Region</div>'
        f'  <div style="font-size:0.85rem; font-weight:700; color:#b8956a;">{chosen["region_acc"]}%</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -- Sample gallery --------------------------------------------------------
    st.sidebar.markdown(
        '<p class="section-label" style="margin-top:1.6rem;">Try a sample</p>',
        unsafe_allow_html=True,
    )
    sample_dishes = [
        "biryani", "butter_chicken", "jalebi",
        "palak_paneer", "daal_baati_churma", "modak",
    ]
    cols = st.sidebar.columns(3)
    for i, dish in enumerate(sample_dishes):
        dish_dir = IMG_DIR / dish
        if dish_dir.exists():
            imgs = sorted(os.listdir(dish_dir))
            if imgs:
                dn = df[df["folder_name"] == dish]["display_name"].values
                name = dn[0] if len(dn) else dish
                with cols[i % 3]:
                    if st.button(name[:10], key=f"s_{dish}", use_container_width=True):
                        st.session_state["sample_image"] = str(dish_dir / imgs[0])

    return chosen["type"]


# -- Main ----------------------------------------------------------------------
def main():
    df, nutrition, idx_to_class, dish_to_region = load_metadata()

    model_type = render_sidebar(df)
    if model_type is None:
        st.stop()

    bundle, device = get_model(model_type)
    if bundle is None:
        st.error("Model weights not found.")
        st.stop()

    # -- Header ----------------------------------------------------------------
    st.markdown(
        '<div class="app-header">'
        '  <p class="app-title">'
        '    <span class="app-title-accent">🍛</span> Regional Thali Identifier'
        '  </p>'
        '  <p class="app-sub">'
        '    Upload an Indian food photo &mdash; identify the dish, region, nutrition and allergens'
        '  </p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -- Upload ----------------------------------------------------------------
    uploaded = st.file_uploader(
        "Drop a food photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    image = None
    if uploaded:
        image = Image.open(uploaded)
    elif "sample_image" in st.session_state:
        image = Image.open(st.session_state["sample_image"])

    if image is None:
        st.markdown(
            '<div class="welcome">'
            '  <div class="welcome-icon">📸</div>'
            '  <h3 style="margin:0.2rem 0 0.5rem; font-weight:600; font-size:1.1rem;">'
            '    Drop a photo above or pick a sample</h3>'
            '  <p style="color:#505058 !important; font-size:0.85rem; margin:0;">'
            '    Supports JPG and PNG</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # -- Predict ---------------------------------------------------------------
    with st.spinner("Analysing..."):
        results = predict(
            image, bundle, device, idx_to_class, dish_to_region, nutrition, df
        )
    top = results[0]

    # -- Two-column layout -----------------------------------------------------
    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        st.image(image, use_container_width=True, output_format="JPEG")

    with col_right:
        rc = REGION_COLORS.get(top["region"], "#78787e")
        re_emoji = REGION_EMOJI.get(top["region"], "🍽️")

        # Top prediction card
        conf_pct = top["confidence"] * 100
        st.markdown(
            '<div class="card">'
            '  <div style="font-size:0.62rem; color:#505058; text-transform:uppercase;'
            '   letter-spacing:0.1em; margin-bottom:0.4rem;">Prediction</div>'
            f'  <div style="font-size:1.5rem; font-weight:700; color:#d0d0d0;'
            f'   line-height:1.2; margin-bottom:0.5rem;">{top["name"]}</div>'
            f'  <span class="region-badge"'
            f'   style="background:{rc}18; color:{rc}; border:1px solid {rc}35;">'
            f'    {re_emoji} {top["region"]} India &middot; {top["state"]}'
            f'  </span>'
            f'  <div style="margin-top:0.7rem;">'
            f'    <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">'
            f'      <span style="font-size:0.72rem; color:#505058;">Confidence</span>'
            f'      <span style="font-size:0.72rem; font-weight:600; color:#909090;">{conf_pct:.1f}%</span>'
            f'    </div>'
            f'    <div class="conf-bg">'
            f'      <div class="conf-fill" style="width:{conf_pct:.1f}%;"></div>'
            f'    </div>'
            f'  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # -- Info grid: Calories + Diet + Allergens ----------------------------
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

            # Diet / course / flavor pills
            diet = top["diet"].lower()
            dcls = "pill-veg" if "veg" in diet and "non" not in diet else "pill-nveg"
            st.markdown(
                '<div style="margin-top:0.5rem;">'
                f'  <span class="pill {dcls}">{top["diet"].title()}</span>'
                f'  <span class="pill">{top["course"].title()}</span>'
                f'  <span class="pill">{top["flavor"].title()}</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        with c2:
            # Allergens
            st.markdown(
                '<div style="font-size:0.62rem; color:#505058; text-transform:uppercase;'
                ' letter-spacing:0.1em; margin-bottom:0.4rem;">Allergens</div>',
                unsafe_allow_html=True,
            )
            amap = {"nuts": "🥜 Nuts", "dairy": "🥛 Dairy", "gluten": "🌾 Gluten"}
            allergens = top.get("allergens", {})
            for key, label in amap.items():
                present = allergens.get(key, False)
                cls = "allergen-yes" if present else "allergen-no"
                txt = "Contains" if present else "Free"
                st.markdown(
                    f'<span class="allergen-badge {cls}">{label} &middot; {txt}</span>',
                    unsafe_allow_html=True,
                )

            # Prep/cook time
            st.markdown(
                f'<div style="margin-top:0.7rem; font-size:0.75rem; color:#505058;">'
                f'  Prep {top["prep_time"]} min &middot; Cook {top["cook_time"]} min'
                f'</div>',
                unsafe_allow_html=True,
            )

        # -- Other candidates --------------------------------------------------
        if len(results) > 1:
            st.markdown(
                '<p class="section-label" style="margin-top:1rem;">Other candidates</p>',
                unsafe_allow_html=True,
            )
            for rank, r in enumerate(results[1:], 2):
                pct = r["confidence"] * 100
                st.markdown(
                    '<div style="display:flex; align-items:center; gap:0.6rem;'
                    ' padding:0.4rem 0; border-bottom:1px solid #1a1a1e;">'
                    f'  <span style="font-size:0.7rem; font-weight:700; color:#404048;'
                    f'   min-width:18px; text-align:center;">#{rank}</span>'
                    f'  <div style="flex:1;">'
                    f'    <span style="font-size:0.82rem; font-weight:600; color:#b0b0b0;">'
                    f'      {r["name"]}</span>'
                    f'    <span style="font-size:0.7rem; color:#505058; margin-left:0.5rem;">'
                    f'      {r["region"]}</span>'
                    f'  </div>'
                    f'  <span style="font-size:0.75rem; font-weight:600; color:#78787e;">'
                    f'    {pct:.1f}%</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # -- Ingredients expander ----------------------------------------------
        with st.expander("Ingredients", expanded=False):
            st.markdown(
                f'<p style="font-size:0.85rem; color:#909090 !important;'
                f' line-height:1.7;">{top["ingredients"]}</p>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
