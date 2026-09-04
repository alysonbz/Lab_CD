
from src.utils import load_wine_dataset
from sklearn.preprocessing import StandardScaler
wine = load_wine_dataset()
X = wine.drop(['Quality'],axis=1)

# Create the scaler
scaler = StandardScaler()

X_norm = X.apply(scaler.fit_transform(X))
print(X_norm.shape)

print('variancia',X.var())

print('variancia do dataset normalizado',X_norm.var())
