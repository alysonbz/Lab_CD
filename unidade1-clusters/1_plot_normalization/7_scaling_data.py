import pandas as pd

from src.utils import load_wine_dataset
from sklearn.preprocessing import StandardScaler
wine = load_wine_dataset()
X = wine.drop(['Quality'],axis=1)

# Create the scaler
scaler = StandardScaler()

X_norm = scaler.fit_transform(X)
X_norm_columns = pd.DataFrame(scaler.fit_transform(X),
                      columns=X.columns)

print('variancia:\n',X.var(),'\n')

print('variancia do dataset normalizado:\n',X_norm_columns.var())
