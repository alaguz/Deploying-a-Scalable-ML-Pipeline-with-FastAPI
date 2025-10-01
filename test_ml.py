import numpy as np
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    load_model,
)


def test_compute_model_metrics_perfect_prediction():
    """compute_model_metrics should return perfect scores when preds == y."""
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_train_and_inference_basic():
    """A simple LogisticRegression should train and predict 0/1 labels."""
    # Linearly separable toy data
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1])

    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == (4,)
    # predictions must be binary 0/1
    assert set(np.unique(preds)).issubset({0, 1})


def test_save_and_load_roundtrip(tmp_path):
    """A saved model should load back and produce identical predictions."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    model = train_model(X, y)

    path = tmp_path / "model.pkl"
    save_model(model, path.as_posix())

    reloaded = load_model(path.as_posix())
    orig_preds = inference(model, X)
    new_preds = inference(reloaded, X)
    assert np.array_equal(orig_preds, new_preds)
