import numpy as np
from scipy.sparse import issparse
from imblearn.over_sampling import SMOTE

def handle_imbalance_smote(X, y, target_class=0, n_synthetic=None):
    """
    Aplica balanceamento sintético via SMOTE utilizando a biblioteca oficial 
    imbalanced-learn de forma otimizada para NLP.
    """
    print("[INFO] Aplicando balanceamento via SMOTE (imblearn)...")
    
    try:
        smote = SMOTE(k_neighbors=3, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"[SUCESSO] SMOTE concluído. Total de amostras após reamostragem: {len(y_resampled)}")
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"[ERRO CRÍTICO] Falha ao executar o SMOTE oficial: {e}")
        raise e