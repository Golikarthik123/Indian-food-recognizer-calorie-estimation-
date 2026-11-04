# Indian Food Recognizer

A Streamlit web application that predicts an Indian food category from an
uploaded image using a fine-tuned MobileNetV2 image-classification model.

## Features

- Upload JPG, JPEG or PNG food images.
- Predict one of the 20 food categories.
- Display the predicted class and its model probability.
- Display the top-three predicted classes.
- Cache the trained model so it is not loaded again on every interaction.

## Model and label files

The following files must be present in the repository root:

- `indian_food_recognizer_finetuned_final.keras` — trained fine-tuned model.
- `class_names.pkl` — the 20 class labels in the exact order used during training.

Do not rename these files unless you also update the paths in `app.py`.

## Run locally

Use Python 3.12.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push the project files to GitHub.
2. In Streamlit Community Cloud, create or open the app connected to this repository.
3. Set the branch to `main` and the main file path to `app.py`.
4. In the app's deployment settings, select **Python 3.12**.
5. Deploy or reboot the app.

The repository root must contain `app.py`, `requirements.txt`, the model file,
and `class_names.pkl`.

## Model preprocessing

Input images are converted to RGB, resized to 224 × 224, converted to
`float32`, and preprocessed using
`tf.keras.applications.mobilenet_v2.preprocess_input`, matching the
MobileNetV2 preprocessing used by the application.

## Supported food classes

burger, butter_naan, chai, chapati, chole_bhature, dal_makhani, dhokla,
fried_rice, idli, jalebi, kaathi_rolls, kadai_paneer, kulfi, masala_dosa,
momos, paani_puri, pakode, pav_bhaji, pizza, samosa.

## Important note

The displayed confidence is the model's predicted class probability. It should
not be interpreted as a guarantee that a prediction is correct.
