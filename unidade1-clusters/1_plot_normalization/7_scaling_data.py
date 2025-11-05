import pandas as pd

from src.utils import load_wine_dataset
from sklearn.preprocessing import StandardScaler
wine = load_wine_dataset()
X = wine.drop(['Quality'],axis=1)

# Create the scaler
scaler = StandardScaler()
wine_scaled = pd.DataFrame(scaler.fit_transform(wine), columns=wine.columns)

X_norm = wine_scaled

print('variancia',X.var)

print('variancia do dataset normalizado',X_norm.var)