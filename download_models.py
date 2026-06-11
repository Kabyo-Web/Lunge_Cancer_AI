import gdown
import os

def download_models():
    os.makedirs("models", exist_ok=True)

    files = {
        "models/feature_extractor.keras": "1oaA2wrRoezE6-E5TAjwcrA3oTmyP_2Ud",
        "models/svm.pkl": "1uzInNU_hy09VRjEr9HDEKyWkwuYzg6Gw",
        "models/scaler.pkl": "17FrqloPyILAlgsP4K3DAUxRSxVF2T1AA",
    }

    for path, file_id in files.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)