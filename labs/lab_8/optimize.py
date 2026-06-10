# Entrena modelos XGBoost para predecir potabilidad del agua, registra cada trial en MLflow y serializa el mejor modelo
# en models/xgboost_model.pkl.

from __future__ import annotations

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import json
import pickle
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_TRIALS = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "water_potability.csv"
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"

TARGET_COL = "Potability"

FEATURE_COLS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]


def save_library_versions(output_dir: Path) -> None:
    packages = [
        "pandas",
        "scikit-learn",
        "xgboost",
        "optuna",
        "mlflow",
    ]

    versions = {}

    for package in packages:
        versions[package] = version(package)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "library_versions.json", "w", encoding="utf-8") as file:
        json.dump(versions, file, indent=4, ensure_ascii=False)


def get_best_model(experiment_id: str):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        raise ValueError("No existen runs registrados para este experimento.")

    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]

    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")

    return best_model


def optimize_model(n_trials: int = N_TRIALS):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {DATA_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    save_library_versions(MODELS_DIR)

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"water_potability_xgboost_optuna_{timestamp}"

    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        run_name = f"XGBoost_trial_{trial.number}_lr_{params['learning_rate']:.4f}_depth_{params['max_depth']}"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        XGBClassifier(
                            **params,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)
            valid_f1 = f1_score(y_valid, y_pred)

            mlflow.log_params(params)
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("imputer", "SimpleImputer_median")
            mlflow.log_metric("valid_f1", valid_f1)

            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(MODELS_DIR / "library_versions.json"))

        return valid_f1

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_water_potability_optimization",
    )

    study.optimize(objective, n_trials=n_trials)

    best_model = get_best_model(experiment_id)

    model_path = MODELS_DIR / "xgboost_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    print("Optimización finalizada.")
    print(f"Experimento MLflow: {experiment_name}")
    print(f"Mejor valid_f1: {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")
    print(f"Modelo serializado en: {model_path}")
    print(f"Versiones guardadas en: {MODELS_DIR / 'library_versions.json'}")

    return best_model


if __name__ == "__main__":
    optimize_model()
