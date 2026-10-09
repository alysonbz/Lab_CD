import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def handle_imbalance_smote(X, y, target_class=0, n_synthetic=None):
    """
    Aplica balanceamento sintético via SMOTE de forma segura e otimizada
    para evitar estouro de memória (RAM) e travamento do computador.
    """
    print("[INFO] Aplicando balanceamento seguro via SMOTE (otimizado para economia de RAM)...")
    
    # Limita o número de amostras sintéticas se for excessivo para poupar hardware
    if n_synthetic is not None and n_synthetic > 10000:
        print(f"[AVISO] Quantidade alta de amostras ({n_synthetic}). Limitando a 5000 para proteger a RAM.")
        n_synthetic = 5000

    try:
        if not SMOTE_AVAILABLE:
            raise ImportError("imblearn não instalado")
        
        # k_neighbors=3 reduz drasticamente o esforço computacional em relação ao padrão (5)
        smote = SMOTE(k_neighbors=3, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
    except Exception as e:
        print(f"[AVISO] SMOTE tradicional falhou/indisponível ({e}). Usando fallback leve...")
        
        if issparse(X):
            X_target = X.tocsr()[y == target_class]
            n_target = X_target.shape[0]
        else:
            X_target = X[y == target_class]
            n_target = len(X_target)
            
        k = min(2, n_target - 1) if n_target > 2 else 1
        nn = NearestNeighbors(n_neighbors=max(1, k), algorithm='ball_tree').fit(X_target)
        
        synthetic = []
        actual_synthetic = min(n_synthetic or 1000, 5000)
        
        for _ in range(actual_synthetic):
            idx = np.random.randint(0, n_target)
            sample = X_target[idx]
            if issparse(sample):
                sample = sample.toarray()[0]
                
            if n_target > 1:
                neighbors = nn.kneighbors([sample], return_distance=False)[0]
                neighbor = X_target[neighbors[1] if len(neighbors) > 1 else neighbors[0]]
                if issparse(neighbor):
                    neighbor = neighbor.toarray()[0]
                diff = neighbor - sample
                synthetic.append(sample + np.random.uniform(0, 1) * diff)
            else:
                synthetic.append(sample)
                
        synthetic_array = np.array(synthetic)
        if issparse(X):
            from scipy.sparse import vstack, csr_matrix
            X_resampled = vstack([X, csr_matrix(synthetic_array)])
        else:
            X_resampled = np.vstack([X, synthetic_array])
            
        y_resampled = np.hstack([y, [target_class] * len(synthetic_array)])
        
    return X_resampled, y_resampled