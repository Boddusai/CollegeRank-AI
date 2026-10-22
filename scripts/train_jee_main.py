from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

DATA = Path("data/processed/jee_main_clean.csv")
MODEL_DIR = Path("backend/ml_models")


def build_examples(df):
    # Construct labels from historical cutoff ranges.
    # Positive: synthetic student rank is at/inside the observed closing rank.
    # Negative: synthetic rank is 1.25x the closing rank.
    # These are derived training examples, not claimed admission records.
    rows = []
    for _, r in df.iterrows():
        cutoff = float(r["closing_rank"])
        base = r.to_dict()

        p = base.copy()
        p["student_rank"] = max(1, int(round(cutoff * 0.9)))
        p["admitted_match"] = 1
        rows.append(p)

        n = base.copy()
        n["student_rank"] = max(1, int(round(cutoff * 1.25)))
        n["admitted_match"] = 0
        rows.append(n)

    return pd.DataFrame(rows)


def main():
    if not DATA.exists():
        raise FileNotFoundError("Run scripts/prepare_jee_main.py first.")

    df = pd.read_csv(DATA)
    examples = build_examples(df)

    train = examples[examples.year <= 2023]
    val = examples[examples.year == 2024]
    test = examples[examples.year == 2025]

    if train.empty or val.empty or test.empty:
        raise ValueError("Need data for 2021-2023 train, 2024 validation, and 2025 test.")

    target = "admitted_match"
    features = [
        "student_rank", "opening_rank", "closing_rank", "year", "round",
        "college", "branch", "category", "gender", "quota"
    ]
    cat = ["college", "branch", "category", "gender", "quota"]
    num = [c for c in features if c not in cat]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42
        )
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        pipe.fit(train[features], train[target])

        p = pipe.predict(val[features])
        prob = pipe.predict_proba(val[features])[:, 1]

        results[name] = {
            "accuracy": float(accuracy_score(val[target], p)),
            "precision": float(precision_score(val[target], p, zero_division=0)),
            "recall": float(recall_score(val[target], p, zero_division=0)),
            "f1": float(f1_score(val[target], p, zero_division=0)),
            "roc_auc": float(roc_auc_score(val[target], prob)),
        }
        fitted[name] = pipe

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = fitted[best_name]

    # Refit selected model on train + validation before final 2025 test.
    train_val = pd.concat([train, val], ignore_index=True)
    best.fit(train_val[features], train_val[target])

    test_prob = best.predict_proba(test[features])[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    results["test_2025"] = {
        "selected_model": best_name,
        "accuracy": float(accuracy_score(test[target], test_pred)),
        "precision": float(precision_score(test[target], test_pred, zero_division=0)),
        "recall": float(recall_score(test[target], test_pred, zero_division=0)),
        "f1": float(f1_score(test[target], test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(test[target], test_prob)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "jee_main_final_model.joblib")
    (MODEL_DIR / "jee_main_metrics.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"Selected model: {best_name}")


if __name__ == "__main__":
    main()
