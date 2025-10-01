# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Task:** Binary classification of income (`>50K` vs `<=50K`)
- **Algorithm:** Logistic Regression (`max_iter=1000`)
- **Preprocessing:** OneHotEncoder (`handle_unknown="ignore"`, dense) for categorical features; LabelBinarizer for target
- **Code:** `ml/model.py`, `ml/data.py`, pipeline in `train_model.py`
- **Artifacts:** `model/model.pkl`, `model/encoder.pkl`
- **Serving (planned):** FastAPI endpoints in `main.py` (`GET /`, `POST /inference`)
- **Notes:** Training emitted a convergence warning; consider higher `max_iter`, feature scaling, or a different solver if tuning.

## Intended Use
- **Primary use:** Educational ML DevOps pipeline demo (training, testing, slice metrics, API).
- **Not intended for:** High-stakes or real-world decisions without rigorous validation, fairness assessment, and governance.

## Training Data
- **File:** `data/census.csv` (Adult/Census Income dataset)
- **Target column:** `salary` (`>50K`, `<=50K`)
- **Categorical features:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- **Handling of missing-like values:** `"?"` kept as its own category.
- **Split:** 80% train / 20% test (stratified by `salary`)
- **Fitting:** `process_data(..., training=True)` fits the encoder and label binarizer on train only.

## Evaluation Data
- **Source:** Held-out 20% test split from the same CSV (stratified by `salary`)
- **Processing:** `process_data(..., training=False, encoder, lb)` reuses the **train-fitted** encoder and label binarizer (no leakage).
- **Inference:** Predictions produced with `inference(model, X_test)`.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

**Global test performance**
- **Precision:** `0.6940`  
- **Recall:** `0.5886`  
- **F1-score:** `0.6370`

**Slice performance (examples)**  
Computed by `performance_on_categorical_slice` and saved in `slice_output.txt` (202 lines total). Representative rows:

| Feature (Value)               | Count | Precision | Recall | F1     |
|------------------------------|------:|----------:|-------:|-------:|
| workclass: `?`               |   363 |   0.6296  | 0.4722 | 0.5397 |
| workclass: `Federal-gov`     |   195 |   0.6377  | 0.5789 | 0.6069 |
| workclass: `Local-gov`       |   389 |   0.6899  | 0.7417 | 0.7149 |
| workclass: `Private`         |  4,597|   0.7307  | 0.5243 | 0.6105 |
| workclass: `Self-emp-inc`    |   222 |   0.7015  | 0.7966 | 0.7460 |
| workclass: `Self-emp-not-inc`|   500 |   0.5930  | 0.7133 | 0.6476 |
| workclass: `State-gov`       |   245 |   0.6400  | 0.7273 | 0.6809 |
| education: `10th`            |   204 |   0.3000  | 0.2000 | 0.2400 |

> Small slices (e.g., count = 1) can show perfect metrics; focus on larger slices for meaningful insights.

## Ethical Considerations
- Potential disparate performance across demographic attributes (`sex`, `race`, `native-country`, etc.).
- Slice reporting is included to reveal disparities; do not deploy without bias analysis, remediation plans, and stakeholder review.

## Caveats and Recommendations
- Baseline linear model; no hyperparameter tuning or feature scaling performed. Consider increasing `max_iter`, trying `liblinear`/`saga`, or scaling numeric features.
- `"?"` treated as a category; alternative cleaning strategies may change results.
- Monitor global and slice metrics over time; re-train if data distribution shifts. Version code, data snapshot, and artifacts.
