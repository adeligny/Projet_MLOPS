import argparse, os, yaml, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_model(name, params):
    if name == "logistic_regression":
        return LogisticRegression(C=float(params.get("C", 1.0)), max_iter=int(params.get("max_iter", 200)))
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", 6)),
            min_samples_split=int(params.get("min_samples_split", 2)),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 8)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model {name}")

def main(args):
    # 1) charge config
    with open("src/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target = cfg["target"]
    num_cols = cfg["features"]["numerical"] or []
    cat_cols = cfg["features"]["categorical"] or []

    # 2) charge CSV
    df = pd.read_csv(args.data)
    # vérifs bavardes
    missing = [c for c in (num_cols + cat_cols + [target]) if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}\nColonnes dispo: {list(df.columns)}")

    # 3) coercition des numériques (si colonnes importées en texte)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) drop des lignes inexploitables
    df = df.dropna(subset=num_cols + cat_cols + [target]).reset_index(drop=True)

    # 5) X/y
    X = df[num_cols + cat_cols]
    y = df[target]

    # 6) split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 42), stratify=y if y.nunique() <= 10 else None)

    # 7) pipeline
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="drop")
    model = get_model(args.model, cfg["models"][args.model])
    pipe = Pipeline([("pre", pre), ("model", model)])

    # 8) MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(args.model)
    with mlflow.start_run(run_name=f"{args.model}_run"):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)) if y.nunique()==2 else None,
        }
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass

        # log
        mlflow.log_metrics({k:v for k,v in metrics.items() if v is not None})
        mlflow.sklearn.log_model(pipe, "model")

        print("Shapes train/test:", X_train.shape, X_test.shape)
        print("Target balance:", y_train.value_counts(normalize=True).to_dict())
        print("Run done:", metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["logistic_regression","decision_tree","random_forest"])
    p.add_argument("--data", required=True)
    main(p.parse_args())
