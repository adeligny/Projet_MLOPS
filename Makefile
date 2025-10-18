.PHONY: setup train app lint test

setup:
    python -m venv .venv && . .venv/Scripts/activate && pip install -r requirements.txt

train:
    python src/train.py --model logistic_regression --data data/raw/loans.csv
    python src/train.py --model decision_tree       --data data/raw/loans.csv
    python src/train.py --model random_forest       --data data/raw/loans.csv

app:
    streamlit run app/streamlit_app.py

lint:
    flake8 src app tests

test:
    pytest -q
