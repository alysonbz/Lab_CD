import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def handle_imbalance_smote(X, y, target_class=0, n_synthetic=2):
    """Aplica balanceamento sintético via SMOTE no espaço de features."""
    print("[INFO] Aplicando balanceamento via SMOTE no espaço de features...")
    try:
        if not SMOTE_AVAILABLE:
            raise ImportError("imblearn não instalado")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except Exception:
        X_target = X[y == target_class]
        k = min(1, len(X_target) - 1) if len(X_target) > 1 else 1
        nn = NearestNeighbors(n_neighbors=max(1, k)).fit(X_target)
        synthetic = []
        for _ in range(n_synthetic):
            idx = np.random.randint(0, len(X_target))
            sample = X_target[idx]
            if len(X_target) > 1:
                neighbors = nn.kneighbors([sample], return_distance=False)[0]
                neighbor = X_target[neighbors[1] if len(neighbors) > 1 else neighbors[0]]
                diff = neighbor - sample
                synthetic.append(sample + np.random.uniform(0, 1) * diff)
            else:
                synthetic.append(sample)
        X_resampled = np.vstack([X, np.array(synthetic)])
        y_resampled = np.hstack([y, [target_class] * n_synthetic])
        
    return X_resampled, y_resampled