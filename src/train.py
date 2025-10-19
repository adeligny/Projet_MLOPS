import argparse
import yaml
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ------------------------------
# 🔧 Chargement de la configuration
# ------------------------------
def load_config(cfg_path="src/config.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------
# 🚀 Fonction principale d’entraînement
# ------------------------------
def main(args):
    cfg = load_config()

    # Lecture des données
    df = pd.read_csv(args.data)
    target = cfg["target"]
    num_cols = cfg["features"]["numerical"]
    cat_cols = cfg["features"]["categorical"]

    X = df[num_cols + cat_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42),
        stratify=y,
    )

    # Prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Sélection du modèle
    model_name = args.model
    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=cfg["models"]["logistic_regression"]["C"],
            max_iter=cfg["models"]["logistic_regression"]["max_iter"],
        )
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=cfg["models"]["decision_tree"]["max_depth"],
            min_samples_split=cfg["models"]["decision_tree"]["min_samples_split"],
            random_state=cfg.get("random_state", 42),
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=cfg["models"]["random_forest"]["n_estimators"],
            max_depth=cfg["models"]["random_forest"]["max_depth"],
            min_samples_split=cfg["models"]["random_forest"]["min_samples_split"],
            random_state=cfg.get("random_state", 42),
        )
    else:
        raise ValueError(f"❌ Modèle non reconnu : {model_name}")

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # ------------------------------
    # 🧪 Tracking avec MLflow
    # ------------------------------
    mlflow.set_experiment(model_name)

    with mlflow.start_run(run_name=f"{model_name}_run"):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")

        mlflow.log_params(cfg["models"][model_name])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(pipe, "model")

    print("✅ Entraînement terminé et enregistré dans MLflow !")

    # ------------------------------
    # 🏆 Export automatique du meilleur modèle
    # ------------------------------
    export_best_model(experiment_name=model_name, metric="f1", mode="max")


# ------------------------------
# 🥇 Sélection et export du meilleur modèle MLflow
# ------------------------------
from mlflow.tracking import MlflowClient
from joblib import dump
import shutil


def export_best_model(experiment_name: str, metric: str = "f1", mode: str = "max"):
    """
    Sélectionne le meilleur run MLflow et exporte son modèle dans models/model.joblib
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"❌ Aucune expérience trouvée : {experiment_name}")
        return

    # Récupérer les runs terminés et trier selon la métrique
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
        max_results=1,
    )

    if not runs:
        print(f"❌ Aucun run terminé trouvé dans {experiment_name}")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id
    best_value = best_run.data.metrics.get(metric, None)
    print(f"🏆 Meilleur modèle : run_id={run_id} | {metric}={best_value:.4f}")

    # Télécharger le modèle depuis MLflow
    os.makedirs("models", exist_ok=True)
    dst_path = client.download_artifacts(run_id, "model", dst_path="models_tmp")

    # Copier le fichier .pkl / .joblib vers models/model.joblib
    for root, _, files in os.walk(dst_path):
        for f in files:
            if f.endswith(".pkl") or f.endswith(".joblib"):
                shutil.copy(os.path.join(root, f), "models/model.joblib")
                shutil.rmtree("models_tmp", ignore_errors=True)
                print("✅ Modèle exporté vers models/model.joblib")
                return
    print("⚠️ Aucun fichier .pkl/.joblib trouvé.")
    shutil.rmtree("models_tmp", ignore_errors=True)


# ------------------------------
# ⚙️ Entrée du script
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    main(args)
