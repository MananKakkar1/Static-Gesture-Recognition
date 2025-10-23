import os
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def load_data(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    X = np.load(Path(data_dir) / "X.npy")
    y = np.load(Path(data_dir) / "y.npy")
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, seed: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=400, 
                                  max_depth=None, 
                                  n_jobs=-1, 
                                  random_state=seed
                                  )

def train(model: RandomForestClassifier, X_train: np.ndarray, y_train: np.ndarray):
    model.fit(X_train, y_train)
    return model

def evaluate(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, name: str = "val") -> dict:
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    print(f"[{name}] accuracy={acc:.3f} macro_f1={f1:.3f}")
    return {"acc": acc, "macro_f1": f1}


def save(model: RandomForestClassifier,
         out_dir: str = "models",
         feature_count: int | None = None,
         label_map: dict | None = None,
         meta_extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = out / f"random_forest.joblib"
    joblib.dump(model, model_path)
    meta = {
        "feature_count": feature_count,
        "label_map": label_map,
        "saved_at": ts,
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(out / f"random_forest.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return model_path


def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    model = build_model()
    train(model, X_train, y_train)
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "validate")
    evaluate(model, X_test, y_test, "test")
    return save(model, 
         feature_count=X.shape[1], 
         label_map={0:"open",1:"fist",2:"thumbs_up",3:"peace"}
         )

if __name__ == "__main__":
    main()

