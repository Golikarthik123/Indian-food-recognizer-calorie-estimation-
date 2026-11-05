from io import BytesIO
from pathlib import Path
import csv
import pickle

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# Streamlit page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Indian Food Recognizer",
    page_icon="🍛",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "indian_food_recognizer_finetuned_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.pkl"
NUTRITION_PATH = BASE_DIR / "nutrition_mapping.csv"
IMAGE_SIZE = (224, 224)
EXPECTED_NUM_CLASSES = 20

CORE_NUTRIENTS = [
    ("Calories", "kcal", "kcal"),
    ("Protein", "g", "protein"),
    ("Carbohydrates", "g", "carbs"),
    ("Total fat", "g", "fat"),
]
EXTRA_NUTRIENTS = [
    ("Dietary fibre", "g", "fibre"),
    ("Free sugars", "g", "sugars"),
    ("Sodium", "mg", "sodium"),
    ("Calcium", "mg", "calcium"),
    ("Iron", "mg", "iron"),
    ("Saturated fat", "g", "saturated_fat"),
]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the trained fine-tuned MobileNetV2 model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")

    model = __import__("tensorflow").keras.models.load_model(MODEL_PATH)
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise ValueError("This app expects one output tensor.")
    if len(output_shape) != 2 or output_shape[-1] != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Unexpected model output shape: {output_shape}. "
            f"Expected (None, {EXPECTED_NUM_CLASSES})."
        )
    return model


@st.cache_data(show_spinner=False)
def load_class_names():
    """Load the class labels saved during model training."""
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class-label file not found: {CLASS_NAMES_PATH.name}")
    with CLASS_NAMES_PATH.open("rb") as file:
        names = pickle.load(file)
    if not isinstance(names, (list, tuple)) or len(names) != EXPECTED_NUM_CLASSES:
        raise ValueError(f"class_names.pkl must contain {EXPECTED_NUM_CLASSES} labels.")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Every class name must be a non-empty string.")
    return list(names)


@st.cache_data(show_spinner=False)
def load_nutrition_data():
    """Load the nutrition mapping CSV and index records by model class."""
    if not NUTRITION_PATH.is_file():
        raise FileNotFoundError(
            "nutrition_mapping.csv is missing. Add it to the repository root."
        )
    with NUTRITION_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    data = {row["class_name"]: row for row in rows}
    if len(data) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected 20 unique nutrition records; found {len(data)}."
        )
    return data


def as_number(row, key):
    """Return a finite numeric CSV value, or None for missing/invalid values."""
    try:
        value = float(row.get(key, ""))
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def predict_food(image, model, class_names):
    """Apply the same MobileNetV2 preprocessing used during training."""
    import tensorflow as tf

    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    predictions = np.asarray(model.predict(np.expand_dims(image_array, axis=0), verbose=0))[0]

    if predictions.ndim != 1 or predictions.shape[0] != len(class_names):
        raise ValueError("Prediction count does not match the class-name count.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("The model returned invalid prediction values.")

    top_indices = np.argsort(predictions)[-3:][::-1]
    return predictions, top_indices


def pretty_food_name(value):
    return value.replace("_", " ").title()


def show_nutrition_estimator(food_class, nutrition_data):
    row = nutrition_data.get(food_class)
    if row is None:
        st.error(f"No nutrition record was found for {pretty_food_name(food_class)}.")
        return

    food_name = row.get("food_name") or pretty_food_name(food_class)
    status = (row.get("status") or "unverified").strip().lower()
    serving_unit = row.get("serving_unit") or "source serving"
    implied_weight = as_number(row, "implied_serving_weight_g")

    st.subheader("🥗 Nutrition estimator")
    st.write(f"**Selected food:** {food_name}")
    st.write(f"**Mapped serving description:** {serving_unit}")

    if status in {"close", "external_close"}:
        st.warning(
            "This is a close food match rather than an exact dish match. "
            "Nutrition values are estimates and may differ from the food you ate."
        )
    elif status == "external":
        st.warning(
            "This nutrition record comes from an external recipe or food source. "
            "It may not represent every version of this dish."
        )

    if implied_weight is not None and implied_weight > 400:
        st.warning(
            f"The mapped source serving implies approximately {implied_weight:.0f} g "
            "and has an unusually high calorie estimate. This may be an oversized "
            "or unsuitable serving definition. Use custom weight if you know the "
            "actual weight; the source value has not been silently changed."
        )

    has_per_100g = all(
        as_number(row, key) is not None
        for key in ["kcal_per_100g", "protein_per_100g", "carbs_per_100g", "fat_per_100g"]
    )

    mode_options = ["Source serving"]
    if has_per_100g:
        mode_options.append("Custom weight (grams)")

    default_mode = (
        "Custom weight (grams)"
        if has_per_100g and implied_weight is not None and implied_weight > 400
        else "Source serving"
    )
    mode = st.radio(
        "How would you like to calculate the portion?",
        mode_options,
        index=mode_options.index(default_mode),
        horizontal=True,
        key=f"portion_mode_{food_class}",
    )

    if mode == "Custom weight (grams)":
        weight_g = st.number_input(
            "Actual food weight (g)",
            min_value=1.0,
            max_value=2000.0,
            value=100.0,
            step=10.0,
            key=f"custom_weight_{food_class}",
            help="Enter the measured weight of the food portion.",
        )
        factor = weight_g / 100.0
        core_values = {
            "kcal": as_number(row, "kcal_per_100g"),
            "protein": as_number(row, "protein_per_100g"),
            "carbs": as_number(row, "carbs_per_100g"),
            "fat": as_number(row, "fat_per_100g"),
        }
        core_values = {
            key: (value * factor if value is not None else None)
            for key, value in core_values.items()
        }
        # Additional nutrients are stored per mapped serving in this CSV.
        # Where the mapped serving weight is available, estimate custom-weight
        # values by scaling those per-serving values by weight.
        # This is an estimate, not an independently measured per-100g value.
        extra_per_serving_keys = {
            "fibre": "fibre_per_serving_g",
            "sugars": "free_sugars_per_serving_g",
            "sodium": "sodium_per_serving_mg",
            "calcium": "calcium_per_serving_mg",
            "iron": "iron_per_serving_mg",
            "saturated_fat": "saturated_fat_per_serving_g",
        }
        extra_values = {}
        if implied_weight is not None and implied_weight > 0:
            weight_factor = weight_g / implied_weight
            for nutrient, csv_key in extra_per_serving_keys.items():
                per_serving_value = as_number(row, csv_key)
                extra_values[nutrient] = (
                    per_serving_value * weight_factor
                    if per_serving_value is not None
                    else None
                )
            st.caption(
                "Main nutrients use the source's per-100g values. Additional "
                "nutrients are estimated by scaling the source's per-serving "
                "values using the mapped serving weight "
                f"({implied_weight:.1f} g). These estimates may be less precise."
            )
        else:
            extra_values = {key: None for _, _, key in EXTRA_NUTRIENTS}
            st.caption(
                "Main nutrients use per-100g values. Additional nutrients are "
                "N/A because this record has no usable serving-weight basis "
                "for scaling its per-serving nutrient values."
            )
        portion_label = f"{weight_g:.0f} g"
        source_scale = None
    else:
        quantity = st.number_input(
            f"Number of servings ({serving_unit})",
            min_value=0.25,
            max_value=20.0,
            value=1.0,
            step=0.25,
            key=f"servings_{food_class}",
        )
        core_keys = {
            "kcal": "kcal_per_serving",
            "protein": "protein_per_serving",
            "carbs": "carbs_per_serving",
            "fat": "fat_per_serving",
        }
        core_values = {
            key: (
                as_number(row, csv_key) * quantity
                if as_number(row, csv_key) is not None
                else None
            )
            for key, csv_key in core_keys.items()
        }
        extra_keys = {
            "fibre": "fibre_per_serving_g",
            "sugars": "free_sugars_per_serving_g",
            "sodium": "sodium_per_serving_mg",
            "calcium": "calcium_per_serving_mg",
            "iron": "iron_per_serving_mg",
            "saturated_fat": "saturated_fat_per_serving_g",
        }
        extra_values = {
            key: (
                as_number(row, csv_key) * quantity
                if as_number(row, csv_key) is not None
                else None
            )
            for key, csv_key in extra_keys.items()
        }
        portion_label = f"{quantity:g} × {serving_unit}"
        source_scale = quantity

    st.markdown(f"**Estimated portion:** {portion_label}")
    st.markdown("#### Main nutrition")
    metric_cols = st.columns(4)
    for col, (label, unit, key) in zip(metric_cols, CORE_NUTRIENTS):
        value = core_values.get(key)
        col.metric(label, f"{value:,.1f} {unit}" if value is not None else "N/A")

    st.markdown("#### Additional nutrition")
    extra_cols = st.columns(3)
    for col, (label, unit, key) in zip(extra_cols, EXTRA_NUTRIENTS):
        value = extra_values.get(key)
        col.metric(label, f"{value:,.1f} {unit}" if value is not None else "N/A")

    source_url = (row.get("source_url") or "").strip()
    primary_source = (row.get("primarysource") or "").strip()
    food_code = (row.get("food_code") or "").strip()
    st.markdown("#### Data source")
    if source_url.startswith(("https://", "http://")):
        st.markdown(f"Source: [{primary_source or 'External source'}]({source_url})")
    else:
        st.write(f"Source database: {primary_source or 'Not specified'}")
        if food_code:
            st.write(f"Food record code: {food_code}")
    st.caption(
        "Nutrition estimates are approximate. Recipe, ingredients, oil, "
        "brand and portion size can change the actual values. This is not "
        "medical or dietary advice."
    )


# ------------------------- Application interface -------------------------

st.title("🍛 Indian Food Recognizer")
st.markdown(
    "Upload a food image or take a photo with your camera. The fine-tuned **MobileNetV2** model predicts "
    "which of the 20 supported food categories it may belong to."
)
st.info(
    "The model suggests the three most likely foods. Please confirm the "
    "correct food before viewing its nutrition estimate."
)

input_method = st.radio(
    "Choose how to provide your food image:",
    ["Upload image", "Use camera"],
    horizontal=True,
    key="image_input_method",
)

if input_method == "Upload image":
    image_file = st.file_uploader(
        "📤 Upload a food image",
        type=["jpg", "jpeg", "png"],
        help="Choose a JPG, JPEG or PNG image.",
        key="food_image_upload",
    )
else:
    image_file = st.camera_input(
        "📷 Take a photo of your food",
        key="food_camera_capture",
    )

if image_file is not None:
    try:
        image_bytes = image_file.getvalue()
        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        st.image(
            image,
            caption="Food image ready for recognition",
            width="stretch",
        )

        if st.button("🔍 Predict Food", type="primary"):
            try:
                with st.spinner("Loading the model and analyzing the image..."):
                    model = load_model()
                    class_names = load_class_names()
                    predictions, top_indices = predict_food(image, model, class_names)
                st.session_state["food_top3"] = [
                    (class_names[int(i)], float(predictions[int(i)]))
                    for i in top_indices
                ]
                st.session_state["food_image_key"] = (
                    f"{input_method}:{getattr(image_file, 'name', 'camera_capture')}"
                )
            except Exception as error:
                st.error("The prediction could not be completed.")
                st.exception(error)

        top3 = st.session_state.get("food_top3")
        if top3:
            st.subheader("🏆 Top-3 predictions")
            for rank, (name, probability) in enumerate(top3, start=1):
                st.write(f"**{rank}. {pretty_food_name(name)}** — {probability * 100:.2f}%")
                st.progress(min(max(probability, 0.0), 1.0))

            st.caption(
                "Confidence is the model's predicted class probability, "
                "not a guarantee that the food is correctly identified."
            )

            nutrition_data = load_nutrition_data()
            options = [name for name, _ in top3]
            selected_food = st.selectbox(
                "Select the food that matches your image:",
                options=options,
                format_func=pretty_food_name,
                key="confirmed_food",
            )
            show_nutrition_estimator(selected_food, nutrition_data)

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The selected image could not be read. Please upload or capture "
                 "a valid JPG, JPEG or PNG image.")

st.divider()
st.caption("Built with Python, TensorFlow, MobileNetV2 and Streamlit.")
