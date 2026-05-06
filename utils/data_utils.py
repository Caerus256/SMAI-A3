"""
Data utilities for T2.6 Regional Thali Identifier.
Provides: metadata loading, name mapping, label lookups, transforms.
"""

import json
import os
import pandas as pd
from pathlib import Path
from torchvision import transforms

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REGION_ORDER = ["North", "South", "West", "East", "North East"]
REGION_COLORS = {
    "North": "#FF9933",
    "South": "#138808",
    "East": "#DAA520",
    "West": "#4169E1",
    "North East": "#8B5CF6",
}
NUM_CLASSES = 80


def load_metadata():
    """Load food_metadata_80.csv and return DataFrame."""
    return pd.read_csv(DATA_DIR / "food_metadata_80.csv")


def load_nutrition():
    """Load nutrition_allergens.json and return dict."""
    with open(DATA_DIR / "nutrition_allergens.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_class_mappings(df=None):
    """Return class_to_idx, idx_to_class, dish_to_region dicts."""
    if df is None:
        df = load_metadata()
    class_names = sorted(df["folder_name"].tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    dish_to_region = dict(zip(df["folder_name"], df["region"]))
    region_to_idx = {r: i for i, r in enumerate(REGION_ORDER)}
    idx_to_region = {i: r for r, i in region_to_idx.items()}
    return {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "dish_to_region": dish_to_region,
        "region_to_idx": region_to_idx,
        "idx_to_region": idx_to_region,
    }


def get_inference_transform():
    """Transform for inference (val/test)."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
