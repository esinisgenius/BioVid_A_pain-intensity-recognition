from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    XGBClassifier = None


def make_baseline_model(name: str, random_state: int) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=3000,
                        C=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "rbf_svc":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        decision_function_shape="ovr",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required for the xgboost baseline")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "xgboost_binary":
        if XGBClassifier is None:
            raise ImportError("xgboost is required for the xgboost baseline")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown baseline model: {name}")
