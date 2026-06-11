# =========================================================
# EfficientNetB0 (TRUNCATED) + GAP+GMP + ML CLASSIFIER (SVM ONLY)
# WITH UNKNOWN CLASS + DATA AUGMENTATION FOR BENIGN
# =========================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics
import sklearn.manifold
import sklearn.svm

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import os
import cv2
import joblib
import random

IMG_SIZE = 128

# =========================================================
# DATASET LOADING
# =========================================================

NORMAL_PATH    = r"dataset/Normal"
BENIGN_PATH    = r"dataset/Benign"
MALIGNANT_PATH = r"dataset/Malignant"
UNKNOWN_PATH   = r"dataset/Unknown"

def load_images(path, limit=None):
    data = []
    files = os.listdir(path)
    if limit:
        files = random.sample(files, min(limit, len(files)))
    for img in files:
        full_path = os.path.join(path, img)
        image = cv2.imread(full_path)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
    return np.array(data)

data_Normal    = load_images(NORMAL_PATH)
data_Benign    = load_images(BENIGN_PATH)
data_Malignant = load_images(MALIGNANT_PATH)
data_Unknown   = load_images(UNKNOWN_PATH, limit=2500)  # Limit Unknown to 2500

print("Normal    :", len(data_Normal))
print("Benign    :", len(data_Benign))
print("Malignant :", len(data_Malignant))
print("Unknown   :", len(data_Unknown))

# =========================================================
# DATA AUGMENTATION FOR BENIGN
# =========================================================

TARGET_BENIGN = 2400  # Increase Benign to match other classes

aug = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

augmented_images = []
needed = TARGET_BENIGN - len(data_Benign)

if needed > 0:
    print(f"\nAugmenting {needed} Benign images...")
    count = 0
    while count < needed:
        for img in data_Benign:
            if count >= needed:
                break
            img_expanded = np.expand_dims(img, axis=0)
            for batch in aug.flow(img_expanded, batch_size=1):
                augmented_images.append(batch[0].astype(np.uint8))
                count += 1
                break

    data_Benign = np.vstack([data_Benign, np.array(augmented_images)])
    print(f"Benign after augmentation: {len(data_Benign)}")

# =========================================================
# PREPARE DATA
# =========================================================

X = preprocess_input(
    np.vstack([
        data_Normal,
        data_Benign,
        data_Malignant,
        data_Unknown
    ]).astype(np.float32)
)

class_names = ['Normal', 'Benign', 'Malignant', 'Unknown']

label_map = {
    'Normal':    0,
    'Benign':    1,
    'Malignant': 2,
    'Unknown':   3
}

y_raw = np.hstack([
    np.array(["Normal"]    * len(data_Normal)),
    np.array(["Benign"]    * len(data_Benign)),
    np.array(["Malignant"] * len(data_Malignant)),
    np.array(["Unknown"]   * len(data_Unknown))
])

y = np.array([label_map[i] for i in y_raw])

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =========================================================
# EFFICIENTNETB0
# =========================================================

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze all layers except last 30
for layer in base_model.layers:
    layer.trainable = False

feature_map = base_model.output
gap = tf.keras.layers.GlobalAveragePooling2D()(feature_map)
gmp = tf.keras.layers.GlobalMaxPooling2D()(feature_map)
features = tf.keras.layers.Concatenate()([gap, gmp])

feature_extractor = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=features
)

# =========================================================
# PARAMETER COUNT
# =========================================================

trainable_params = np.sum(
    [tf.keras.backend.count_params(w) for w in feature_extractor.trainable_weights]
)
non_trainable_params = np.sum(
    [tf.keras.backend.count_params(w) for w in feature_extractor.non_trainable_weights]
)
total_params = trainable_params + non_trainable_params

print("\n========== PARAMETER COUNT ==========")
print(f"Trainable Parameters     : {trainable_params:,}")
print(f"Non-Trainable Parameters : {non_trainable_params:,}")
print(f"Total Parameters         : {total_params:,}")

# =========================================================
# MFLOPs
# =========================================================

def get_flops(model):
    concrete = tf.function(
        lambda x: model(x)
    ).get_concrete_function(
        tf.TensorSpec([1, 128, 128, 3], tf.float32)
    )
    frozen_func = convert_variables_to_constants_v2(concrete)
    graph_def = frozen_func.graph.as_graph_def()
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd='op', options=opts
        )
    return flops.total_float_ops

flops  = get_flops(feature_extractor)
mflops = flops / 1e6

print("\n========== COMPUTATIONAL COST ==========")
print(f"MFLOPs : {mflops:.2f}")

# =========================================================
# FEATURE EXTRACTION
# =========================================================

X_train_feat = feature_extractor.predict(X_train, batch_size=32, verbose=1)
X_test_feat  = feature_extractor.predict(X_test,  batch_size=32, verbose=1)

# =========================================================
# SCALING
# =========================================================

scaler = sklearn.preprocessing.RobustScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_test_feat  = scaler.transform(X_test_feat)

# =========================================================
# SVM WITH CLASS WEIGHT BALANCED
# =========================================================

svm = sklearn.svm.SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    class_weight='balanced'  # Handles class imbalance
)

svm.fit(X_train_feat, y_train)
y_pred = svm.predict(X_test_feat)

# =========================================================
# RESULTS
# =========================================================

print("\n===== SVM RESULTS =====")
print(f"Accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred):.4f}")
print(sklearn.metrics.classification_report(
    y_test, y_pred, target_names=class_names
))

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================================================
# ROC CURVE
# =========================================================

y_test_bin = sklearn.preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3])
y_score    = svm.predict_proba(X_test_feat)

plt.figure(figsize=(7, 6))
for i in range(4):
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
    auc_score   = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} AUC={auc_score:.3f}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("SVM ROC Curve")
plt.legend()
plt.grid()
plt.show()

# =========================================================
# TSNE
# =========================================================

tsne   = sklearn.manifold.TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_test_feat)

plt.figure(figsize=(7, 6))
for i, label in enumerate(class_names):
    idx = y_test == i
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label)

plt.title("t-SNE Visualization (SVM Features)")
plt.legend()
plt.show()

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n========== FINAL SUMMARY ==========")
print(f"Accuracy  : {sklearn.metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Parameters: {total_params:,}")
print(f"MFLOPs    : {mflops:.2f}")

# =========================================================
# SAVE MODELS
# =========================================================

os.makedirs("models", exist_ok=True)

feature_extractor.save("models/feature_extractor.keras")
joblib.dump(svm,    "models/svm.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("All models saved successfully.")