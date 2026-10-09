import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


class PyTorchMLPClassifier(nn.Module):
    """Implementação de uma Rede Neural Multicamadas (MLP) utilizando PyTorch."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 2):
        super(PyTorchMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PyTorchMLPWrapper:
    """Wrapper compatível com o scikit-learn para treinar na GPU, prever e exportar para ONNX."""
    def __init__(self, hidden_dim: int = 128, max_iter: int = 30, random_state: int = 42, device: str = None):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Detecção automática de GPU (CUDA para NVIDIA, MPS para Apple Silicon ou CPU)
        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("[INFO] PyTorch utilizando dispositivo: CUDA (GPU NVIDIA)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("[INFO] PyTorch utilizando dispositivo: MPS (Apple Silicon GPU)")
            else:
                self.device = torch.device("cpu")
                print("[INFO] PyTorch utilizando dispositivo: CPU")

        self.model = None
        self.classes_ = np.array([0, 1])
        torch.manual_seed(random_state)

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        # Envia os tensores para a GPU/device configurado
        X_tensor = torch.tensor(X_dense, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        # Cálculo dinâmico de pesos por classe para balanceamento no PyTorch (CrossEntropyLoss com weight)
        class_counts = np.bincount(y)
        total_samples = len(y)
        weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        input_dim = X_tensor.shape[1]
        self.model = PyTorchMLPClassifier(input_dim=input_dim, hidden_dim=self.hidden_dim).to(self.device)
        
        # Aplica os pesos balanceados na função de perda
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        X_tensor = torch.tensor(X_dense, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def export_to_onnx(self, filepath: str, sample_input_shape: tuple):
        """Exporta o modelo treinado em PyTorch para o formato padrão ONNX."""
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado antes de ser exportado para ONNX.")
        
        try:
            self.model.eval()
            self.model.to("cpu")
            dummy_input = torch.randn(sample_input_shape, dtype=torch.float32)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                filepath,
                export_params=True,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"[SUCESSO] Modelo PyTorch exportado para ONNX em: {filepath}")
        except Exception as e:
            print(f"[AVISO] Falha ao exportar para ONNX (instale 'onnxscript' e 'onnx'): {e}")
        finally:
            if self.model is not None:
                self.model.to(self.device)


class TextClassifier:
    def __init__(self, random_state=42, device=None):
        self.random_state = random_state
        self.device = device

    def get_models(self):
        models = {
            "Naive Bayes": MultinomialNB(),

            "SVM": LinearSVC(
                random_state=self.random_state,
                dual="auto",
                class_weight='balanced'
            ),

            "Regressão Logística": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),

            "MLP": PyTorchMLPWrapper(
                max_iter=30,
                random_state=self.random_state,
                device=self.device
            )
        }

        return models