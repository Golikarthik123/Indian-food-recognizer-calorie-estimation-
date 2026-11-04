from io import BytesIO
from pathlib import Path
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
IMAGE_SIZE = (224, 224)
EXPECTED_NUM_CLASSES = 20


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the trained fine-tuned MobileNetV2 model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Make sure it is committed to the GitHub repository."
        )

    model = __import__("tensorflow").keras.models.load_model(MODEL_PATH)

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise ValueError("This app expects a model with one output tensor.")

    if len(output_shape) != 2 or output_shape[-1] != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Unexpected model output shape: {output_shape}. "
            f"Expected (None, {EXPECTED_NUM_CLASSES})."
        )

    return model


@st.cache_data(show_spinner=False)
def load_class_names():
    """Load and cache the class labels saved during model training."""
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class-label file not found: {CLASS_NAMES_PATH.name}."
        )

    # This pickle file must come from this trusted project repository.
    with CLASS_NAMES_PATH.open("rb") as file:
        names = pickle.load(file)

    if not isinstance(names, (list, tuple)):
        raise TypeError("class_names.pkl must contain a list or tuple.")

    if len(names) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} class names, found {len(names)}."
        )

    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Every class name must be a non-empty string.")

    return list(names)


def predict_food(image, model, class_names):
    """Apply the same MobileNetV2 preprocessing used during training."""
    import tensorflow as tf

    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )
    image_batch = np.expand_dims(image_array, axis=0)

    predictions = np.asarray(model.predict(image_batch, verbose=0))[0]

    if predictions.ndim != 1 or predictions.shape[0] != len(class_names):
        raise ValueError(
            "The model prediction count does not match the class-name count."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError("The model returned invalid prediction values.")

    top_indices = np.argsort(predictions)[-3:][::-1]
    return predictions, top_indices


# ------------------------- Application interface -------------------------

st.title("🍛 Indian Food Recognizer")
st.markdown(
    """
    Upload a food image and the fine-tuned **MobileNetV2** model will
    predict which of the 20 supported food categories it belongs to.
    """
)

st.info(
    "Supported categories include burger, dosa, idli, samosa, "
    "paani puri, biryani-style dishes and more."
)

uploaded_file = st.file_uploader(
    "📤 Upload a food image",
    type=["jpg", "jpeg", "png"],
    help="Choose a JPG, JPEG or PNG image.",
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes))
        image.load()

        st.image(
            image,
            caption="Uploaded food image",
            width="stretch",
        )

        if st.button("🔍 Predict Food", type="primary"):
            try:
                with st.spinner("Loading the model and analyzing the image..."):
                    model = load_model()
                    class_names = load_class_names()
                    predictions, top_indices = predict_food(
                        image, model, class_names
                    )

                predicted_index = int(top_indices[0])
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index])

                st.success(f"Predicted food: **{predicted_class}**")
                st.metric(
                    label="Prediction confidence",
                    value=f"{confidence * 100:.2f}%",
                )

                st.subheader("🏆 Top-3 predictions")
                for rank, index in enumerate(top_indices, start=1):
                    probability = float(predictions[index])
                    st.write(
                        f"**{rank}. {class_names[int(index)]}** — "
                        f"{probability * 100:.2f}%"
                    )
                    st.progress(
                        min(max(probability, 0.0), 1.0)
                    )

                st.caption(
                    "Predictions are generated by the fine-tuned "
                    "MobileNetV2 model. Confidence is the model's "
                    "predicted class probability, not a guarantee."
                )

            except Exception as error:
                st.error(
                    "The prediction could not be completed. "
                    "Check that the model and class_names.pkl files "
                    "are present and compatible."
                )
                st.exception(error)

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The uploaded file could not be read as a valid image.")

st.divider()
st.caption("Built with Python, TensorFlow, MobileNetV2 and Streamlit.")
