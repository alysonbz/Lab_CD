
from src.utils import load_wine_dataset
from ___ import ___
wine = load_wine_dataset()
X = wine.drop(['Quality'],axis=1)

# Create the scaler
scaler = ____

X_norm = _____

print('variancia',__)

print('variancia do dataset normalizado',__)
