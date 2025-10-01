import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Resolve project root and data path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")

print(DATA_PATH)

# Load data
data = pd.read_csv(DATA_PATH)

# Train/test split (use stratify for class balance if label exists)
train, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["salary"]
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process train and test
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Ensure model directory exists and save artifacts
model_dir = os.path.join(PROJECT_PATH, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)

# Load the model (sanity check round-trip)
model = load_model(model_path)

# Inference on test set
preds = inference(model, X_test)

# Calculate and print global metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute and save slice metrics
slice_out_path = os.path.join(PROJECT_PATH, "slice_output.txt")
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p_s, r_s, fb_s = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )
        with open(slice_out_path, "a", encoding="utf-8") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p_s:.4f} | Recall: {r_s:.4f} | F1: {fb_s:.4f}", file=f)
