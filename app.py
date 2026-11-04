
import streamlit as st
print("CHECKPOINT 1: App started", flush=True)

st.set_page_config(
    page_title="Indian Food Recognizer",
    page_icon="🍛",
    layout="centered"
)

print("CHECKPOINT 2: Before TensorFlow import", flush=True)
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from pathlib import Path
print("CHECKPOINT 3: TensorFlow imported", flush=True)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Indian Food Recognizer",
    page_icon="🍛",
    layout="centered"
)

# --------------------------------------------------
# 2. PROJECT FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "indian_food_recognizer_finetuned_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.pkl"


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    print("CHECKPOINT 4: Loading model", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CHECKPOINT 5: Model loaded", flush=True)
    return model


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()
class_names = load_class_names()


# --------------------------------------------------
# 4. FOOD PREDICTION FUNCTION
# --------------------------------------------------

def predict_food(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image into NumPy array
    image_array = np.array(image).astype(np.float32)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)

    # Model prediction
    predictions = model.predict(image_batch, verbose=0)[0]

    # Get Top-3 predictions
    top3_indices = np.argsort(predictions)[-3:][::-1]

    return predictions, top3_indices


# --------------------------------------------------
# 5. STREAMLIT APPLICATION INTERFACE
# --------------------------------------------------

st.title("🍛 Indian Food Recognizer")

st.markdown(
    """
    ### Welcome to Indian Food Recognizer!

    Upload an image of food and our fine-tuned
    **MobileNetV2 Deep Learning model** will identify it.

    **Supported categories:** 20 food classes.
    """
)

st.divider()

# Image uploader
uploaded_file = st.file_uploader(
    "📤 Upload a Food Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 6. DISPLAY IMAGE AND PREDICTIONS
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Food Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Food", type="primary"):

        with st.spinner("Analyzing your food image..."):

            predictions, top3_indices = predict_food(image)

            predicted_class = class_names[top3_indices[0]]

            confidence = float(
                predictions[top3_indices[0]] * 100
            )

        # Final prediction
        st.success(
            f"Predicted Food: {predicted_class}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.divider()

        # Top-3 predictions
        st.subheader("🏆 Top-3 Predictions")

        for rank, idx in enumerate(top3_indices, start=1):

            probability = float(predictions[idx])

            st.write(
                f"**{rank}. {class_names[idx]}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

        st.caption(
            "Predictions are generated using the fine-tuned "
            "MobileNetV2 model."
        )

st.divider()

st.caption(
    "Built with Python, TensorFlow, MobileNetV2 and Streamlit."
)
