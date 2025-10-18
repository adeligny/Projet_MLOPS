import pandas as pd, yaml
df = pd.read_csv('data/raw/loans.csv')
print('Shape:', df.shape)
print('First 3 rows:\\n', df.head(3))
print('\\nDtypes:\\n', df.dtypes)

cfg = yaml.safe_load(open('src/config.yaml', encoding='utf-8'))
num = cfg['features']['numerical']
cat = cfg['features']['categorical']
target = cfg['target']

missing = [c for c in (num + cat + [target]) if c not in df.columns]
print('\\nMissing columns:', missing)

# Aperçu de la cible
print('\\nTarget counts:')
print(df[target].value_counts(dropna=False).head())

# Aperçu de valeurs non-numériques dans les colonnes numériques
for c in num:
    bad = df[c].astype(str).str.contains('[^0-9eE+\\-\\.]', regex=True).sum()
    print(f'Non-numeric chars in {c}:', bad)
