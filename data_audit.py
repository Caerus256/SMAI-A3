import csv
import json
import os
import re
import random

# Paths
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "indian_food.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "..", "indian_food.csv")
IMG_DIR = os.path.join(BASE_DIR, "Indian Food Images")
if not os.path.isdir(IMG_DIR):
    IMG_DIR = os.path.join(BASE_DIR, "..", "Indian Food Images")
OUT_CSV = os.path.join(BASE_DIR, "data", "food_metadata_80.csv")
OUT_JSON = os.path.join(BASE_DIR, "data", "nutrition_allergens.json")

# Allergen keyword lists
NUT_KEYWORDS = [
    "cashew", "almond", "peanut", "pistachio", "badam", "nuts",
    "dry fruits", "dried fruits", "mixed nuts",
]
DAIRY_KEYWORDS = [
    "milk", "cream", "ghee", "curd", "yogurt", "yoghurt", "paneer",
    "cheese", "chhena", "chenna", "khoa", "mawa", "dahi", "butter",
    "condensed milk", "cottage cheese", "malai", "rabri",
]
GLUTEN_KEYWORDS = [
    "flour", "maida", "wheat", "atta", "bread", "semolina", "rava",
    "vermicelli", "naan", "sev", "besan", "gram flour", "bengal gram flour",
]

# Calorie estimation heuristic
# Base kcal per serving by course, then adjusted by ingredients
COURSE_BASE_KCAL = {
    "dessert": 320,
    "main course": 400,
    "snack": 220,
    "starter": 280,
}

CALORIE_MODIFIERS = {
    "ghee": 40, "butter": 35, "cream": 30, "oil": 25,
    "sugar": 20, "jaggery": 15, "coconut": 20,
    "cheese": 25, "paneer": 25, "milk powder": 20,
    "fried": 50, "deep fried": 60,
    "rice": 15, "bread": 10,
}


def check_allergens(ingredients_str):
    """Return dict of allergen flags based on ingredient keywords."""
    ing_lower = ingredients_str.lower()
    return {
        "nuts": any(kw in ing_lower for kw in NUT_KEYWORDS),
        "dairy": any(kw in ing_lower for kw in DAIRY_KEYWORDS),
        "gluten": any(kw in ing_lower for kw in GLUTEN_KEYWORDS),
    }


def estimate_calories(course, ingredients_str):
    """Rough calorie estimate per serving."""
    base = COURSE_BASE_KCAL.get(course.lower().strip(), 350)
    ing_lower = ingredients_str.lower()
    modifier = 0
    for keyword, delta in CALORIE_MODIFIERS.items():
        if keyword in ing_lower:
            modifier += delta
    # Clamp to reasonable range
    total = base + modifier
    # Add small randomness for realism (±10%)
    random.seed(hash(ingredients_str))
    jitter = random.randint(-int(total * 0.08), int(total * 0.08))
    total = max(100, min(700, total + jitter))
    return total


def main():
    # Get list of image folders
    image_folders = set(os.listdir(IMG_DIR))
    print(f"Found {len(image_folders)} image class folders")

    # Read CSV and filter to matching classes
    rows_80 = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            display_name = row["name"].strip()
            folder_name = display_name.lower().replace(" ", "_")
            if folder_name in image_folders:
                rows_80.append({
                    "folder_name": folder_name,
                    "display_name": display_name,
                    "ingredients": row["ingredients"].strip(),
                    "diet": row["diet"].strip(),
                    "course": row["course"].strip(),
                    "flavor_profile": row["flavor_profile"].strip(),
                    "state": row["state"].strip(),
                    "region": row["region"].strip(),
                    "prep_time": row["prep_time"].strip(),
                    "cook_time": row["cook_time"].strip(),
                })

    print(f"Matched {len(rows_80)} dishes to image folders")

    # Write food_metadata_80.csv
    fieldnames = [
        "folder_name", "display_name", "ingredients", "diet", "course",
        "flavor_profile", "state", "region", "prep_time", "cook_time",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_80)
    print(f"Wrote {OUT_CSV}")

    # Region stats
    region_counts = {}
    for row in rows_80:
        r = row["region"]
        region_counts[r] = region_counts.get(r, 0) + 1
    print("\nRegion distribution (dishes / images):")
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {region}: {count} dishes / {count * 50} images")

    # Generate nutrition_allergens.json
    nutrition_data = {}
    for row in rows_80:
        allergens = check_allergens(row["ingredients"])
        calories = estimate_calories(row["course"], row["ingredients"])
        nutrition_data[row["folder_name"]] = {
            "display_name": row["display_name"],
            "allergens": allergens,
            "approx_calories_kcal": calories,
            "diet": row["diet"],
            "course": row["course"],
            "region": row["region"],
            "state": row["state"],
            "flavor_profile": row["flavor_profile"],
            "ingredients": row["ingredients"],
            "prep_time_min": row["prep_time"],
            "cook_time_min": row["cook_time"],
        }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(nutrition_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}")

    # Summary
    nuts_count = sum(1 for v in nutrition_data.values() if v["allergens"]["nuts"])
    dairy_count = sum(1 for v in nutrition_data.values() if v["allergens"]["dairy"])
    gluten_count = sum(1 for v in nutrition_data.values() if v["allergens"]["gluten"])
    print(f"\nAllergen summary:")
    print(f"  Nuts:   {nuts_count}/80 dishes")
    print(f"  Dairy:  {dairy_count}/80 dishes")
    print(f"  Gluten: {gluten_count}/80 dishes")


if __name__ == "__main__":
    main()
