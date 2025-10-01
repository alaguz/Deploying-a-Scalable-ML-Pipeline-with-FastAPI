from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from joblib import dump, load
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Serializes model or encoder to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder/LabelBinarizer.
    path : str
        Path to save the file.
    """
    dump(model, path)


def load_model(path):
    """Loads a serialized object from `path` and returns it."""
    return load(path)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics on a slice of the data specified by a column name
    and a slice value.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`.
    column_name : str
        Column containing the sliced feature.
    slice_value : str | int | float
        Value of the slice feature.
    categorical_features : list
        Names of categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model
        Trained model used for prediction.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    df_slice = data[data[column_name] == slice_value]
    if df_slice.empty:
        # No rows for this slice; return zeros to avoid errors.
        return 0.0, 0.0, 0.0

    X_slice, y_slice, _, _ = process_data(
        df_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
