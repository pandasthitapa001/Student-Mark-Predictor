"""Train and evaluate models using the prepared data.

This module mirrors the notebook's training logic:
- regression: LinearRegression, RandomForestRegressor
- classification: LogisticRegression, DecisionTreeClassifier

It saves the best regression and classification models and a permutation
importance DataFrame into the `artifacts/` folder.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_processing import load_and_prepare_data


def train_and_evaluate(base_dir: Path | str = Path(__file__).resolve().parent):
    base_dir = Path(base_dir)
    prepared = load_and_prepare_data(base_dir)

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train_reg = prepared["y_train_reg"]
    y_test_reg = prepared["y_test_reg"]

    X_train_clf = prepared["X_train_clf"]
    X_test_clf = prepared["X_test_clf"]
    y_train_clf = prepared["y_train_clf"]
    y_test_clf = prepared["y_test_clf"]

    linear_preprocessor = prepared["linear_preprocessor"]
    tree_preprocessor = prepared["tree_preprocessor"]

    # Build models (pipelines)
    from sklearn.pipeline import Pipeline

    linear_regression_model = Pipeline(
        steps=[("preprocessor", linear_preprocessor), ("model", LinearRegression())]
    )

    random_forest_regressor_model = Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor,
            ),
            (
                "model",
                RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2),
            ),
        ]
    )

    logistic_regression_classifier = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    decision_tree_classifier = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=5, min_samples_leaf=5, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    regression_models = {
        "Linear Regression": linear_regression_model,
        "Random Forest Regressor": random_forest_regressor_model,
    }

    classification_models = {
        "Logistic Regression": logistic_regression_classifier,
        "Decision Tree Classifier": decision_tree_classifier,
    }

    # Train regression
    for name, model in regression_models.items():
        model.fit(X_train, y_train_reg)

    # Evaluate regression
    regression_results = []
    for name, model in regression_models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test_reg, preds)
        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
        r2 = r2_score(y_test_reg, preds)

        regression_results.append({"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)})

    regression_results_df = pd.DataFrame(regression_results).sort_values(by="RMSE")
    best_regression_model_name = regression_results_df.iloc[0]["Model"]
    best_regression_model = regression_models[best_regression_model_name]

    # Train classification
    for name, model in classification_models.items():
        model.fit(X_train_clf, y_train_clf)

    # Evaluate classification
    classification_results = []
    for name, model in classification_models.items():
        preds = model.predict(X_test_clf)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_clf)[:, 1]
            roc_auc = roc_auc_score(y_test_clf, probs)
        else:
            roc_auc = np.nan

        classification_results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test_clf, preds), 4),
                "Precision": round(precision_score(y_test_clf, preds, zero_division=0), 4),
                "Recall": round(recall_score(y_test_clf, preds, zero_division=0), 4),
                "F1": round(f1_score(y_test_clf, preds, zero_division=0), 4),
                "ROC_AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
            }
        )

    classification_results_df = pd.DataFrame(classification_results).sort_values(by="F1", ascending=False)
    best_classification_model_name = classification_results_df.iloc[0]["Model"]
    best_classification_model = classification_models[best_classification_model_name]

    # Permutation importance for regression best model
    reg_perm = permutation_importance(
        best_regression_model,
        X_test,
        y_test_reg,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )

    reg_importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": reg_perm.importances_mean,
        "importance_std": reg_perm.importances_std,
    }).sort_values(by="importance_mean", ascending=False)

    # Save artifacts
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(best_regression_model, artifacts_dir / "best_regression_model.joblib")
    joblib.dump(best_classification_model, artifacts_dir / "best_classification_model.joblib")
    joblib.dump(reg_importance_df, artifacts_dir / "regression_feature_importance.joblib")

    return {
        "regression_results_df": regression_results_df,
        "classification_results_df": classification_results_df,
        "best_regression_model_name": best_regression_model_name,
        "best_classification_model_name": best_classification_model_name,
        "artifacts_dir": artifacts_dir,
        "reg_importance_df": reg_importance_df,
    }


if __name__ == "__main__":
    res = train_and_evaluate()
    print("Regression results:\n", res["regression_results_df"])
    print("\nClassification results:\n", res["classification_results_df"])
    print("Artifacts saved to:", res["artifacts_dir"])
