import yaml
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="Loan Default – Prédiction", layout="wide")
st.title("🔮 Loan Default — Interface simple")

# --- Configurations de base ---
TRACKING_URI = "mlruns"      # MLflow local
DATA_PATH = "data/raw/loans.csv"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# --- Lire config.yaml ---
with open("src/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

target = cfg["target"]
num_cols = cfg["features"]["numerical"] or []
cat_cols = cfg["features"]["categorical"] or []
features = [c for c in (num_cols + cat_cols) if c != target]

# --- Récupération des experiments MLflow ---
exp_names = [e.name for e in client.search_experiments()]
if not exp_names:
    st.warning("Aucun experiment MLflow trouvé. Lancez un entraînement.")
    st.stop()

exp_name = st.selectbox("Experiment MLflow", options=sorted(exp_names), index=0)

# --- Dernier run terminé ---
def get_latest_finished_run(experiment_name: str):
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None

run = get_latest_finished_run(exp_name)
if not run:
    st.warning("Aucun run 'FINISHED' pour cet experiment. Lancez un entraînement.")
    st.stop()

run_id = run.info.run_id
st.success(f"Run sélectionné : {run_id}")

# --- Afficher les métriques principales ---
m = run.data.metrics
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{m.get('accuracy', float('nan')):.3f}" if 'accuracy' in m else "—")
col2.metric("F1", f"{m.get('f1', float('nan')):.3f}" if 'f1' in m else "—")
col3.metric("ROC AUC", f"{m.get('roc_auc', float('nan')):.3f}" if 'roc_auc' in m else "—")

# --- Charger le modèle ---
try:
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

# --- Charger le CSV pour valeurs par défaut ---
try:
    df = pd.read_csv(DATA_PATH)
    if target in df.columns:
        df = df.drop(columns=[target])
except Exception:
    df = pd.DataFrame(columns=features)

defaults_num = {}
defaults_cat = {}
choices_cat = {}

if not df.empty:
    for c in num_cols:
        if c in df.columns:
            defaults_num[c] = float(df[c].median())
    for c in cat_cols:
        if c in df.columns:
            mode_val = df[c].mode()
            defaults_cat[c] = mode_val.iloc[0] if not mode_val.empty else ""
            uniq = df[c].dropna().astype(str).unique().tolist()
            choices_cat[c] = uniq[:50] if uniq else [""]
else:
    for c in num_cols:
        defaults_num[c] = 0.0
    for c in cat_cols:
        defaults_cat[c] = ""
        choices_cat[c] = [""]

# --- Formulaire de prédiction unitaire ---
st.subheader("🧾 Remplissez les caractéristiques pour obtenir une prédiction")
with st.form("predict_form"):
    cols = st.columns(3)
    for i, c in enumerate(num_cols):
        with cols[i % 3]:
            st.number_input(c, value=float(defaults_num.get(c, 0.0)), key=f"num_{c}")
    for i, c in enumerate(cat_cols):
        with cols[i % 3]:
            opts = choices_cat.get(c, [""])
            val = defaults_cat.get(c, opts[0] if opts else "")
            st.selectbox(c, options=opts, index=(opts.index(val) if val in opts else 0), key=f"cat_{c}")
    submitted = st.form_submit_button("Prédire")

if submitted:
    # Créer un DataFrame avec une seule ligne à partir des entrées utilisateur
    row = {}
    for c in num_cols:
        row[c] = [st.session_state[f"num_{c}"]]
    for c in cat_cols:
        row[c] = [st.session_state[f"cat_{c}"]]
    X_one = pd.DataFrame(row)[features]

    try:
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(X_one)[:, 1][0])
            st.success(f"Probabilité de défaut estimée : {p:.3f}")
        else:
            yhat = int(model.predict(X_one)[0])
            st.success(f"Prédiction : {yhat}")
    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
