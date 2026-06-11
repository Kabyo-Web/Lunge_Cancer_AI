import cv2
import joblib
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

IMG_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")  # ✅ fix

feature_extractor = load_model(
    os.path.join(MODELS_DIR, "feature_extractor.keras"),
    compile=False
)

svm_model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

classes = ["Normal", "Benign", "Malignant", "Unknown"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or cannot be read!")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))
    features = feature_extractor.predict(img, verbose=0)
    features = np.array(features)
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    prediction = svm_model.predict(features)[0]
    probabilities = svm_model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    if classes[prediction] == "Unknown":
        raise ValueError("NOT_CT_SCAN")

    return classes[prediction], confidence