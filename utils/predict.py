"""
Inference helper for T2.6 Regional Thali Identifier.
Loads model + metadata, predicts dish from a PIL image.
"""

import torch
import timm
from PIL import Image
from .data_utils import (
    load_metadata, load_nutrition, get_class_mappings,
    get_inference_transform, MODEL_DIR, NUM_CLASSES,
)


def load_model(model_path=None, device=None):
    """Load the best EfficientNet-B0 model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = MODEL_DIR / "efficientnet_b0_best.pth"

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, device


def predict_image(image: Image.Image, model, device, top_k=3):
    """
    Predict dish from a PIL image.
    Returns list of top_k dicts: {dish, display_name, region, confidence, nutrition}
    """
    df = load_metadata()
    nutrition = load_nutrition()
    mappings = get_class_mappings(df)
    transform = get_inference_transform()

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        folder_name = mappings["idx_to_class"][idx]
        region = mappings["dish_to_region"][folder_name]
        info = nutrition.get(folder_name, {})
        row = df[df["folder_name"] == folder_name].iloc[0]
        results.append({
            "folder_name": folder_name,
            "display_name": info.get("display_name", row["display_name"]),
            "confidence": prob,
            "region": region,
            "state": info.get("state", row["state"]),
            "diet": info.get("diet", row["diet"]),
            "course": info.get("course", row["course"]),
            "flavor_profile": info.get("flavor_profile", row["flavor_profile"]),
            "ingredients": info.get("ingredients", row["ingredients"]),
            "prep_time": info.get("prep_time_min", str(row["prep_time"])),
            "cook_time": info.get("cook_time_min", str(row["cook_time"])),
            "allergens": info.get("allergens", {}),
            "calories": info.get("approx_calories_kcal", "N/A"),
        })

    return results
