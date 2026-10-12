import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter


class PyTorchBiLSTMClassifier(nn.Module):
    """Arquitetura BiLSTM baseada em PyTorch para classificação de texto."""
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 64, output_dim: int = 2, dropout: float = 0.3):
        super(PyTorchBiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class PyTorchBiLSTMWrapper:
    """Wrapper compatível com o scikit-learn para treinar, prever e exportar para ONNX usando a BiLSTM."""
    def __init__(self, vocab_size: int = 10000, max_length: int = 100, embedding_dim: int = 100, 
                 hidden_dim: int = 64, max_epochs: int = 5, batch_size: int = 64, random_state: int = 42, device: str = None):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("[INFO] BiLSTM utilizando dispositivo: CUDA (GPU NVIDIA)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("[INFO] BiLSTM utilizando dispositivo: MPS (Apple Silicon GPU)")
            else:
                self.device = torch.device("cpu")
                print("[INFO] BiLSTM utilizando dispositivo: CPU")

        self.model = None
        self.word2idx = {}
        self.classes_ = np.array([0, 1])
        torch.manual_seed(random_state)

    def _tokenize_and_pad(self, texts, fit=False):
        if fit:
            word_counts = Counter()
            for text in texts:
                words = str(text).lower().split()
                word_counts.update(words)
            
            most_common = word_counts.most_common(self.vocab_size - 2)
            self.word2idx = {word: idx + 2 for idx, (word, count) in enumerate(most_common)}
            self.word2idx["<PAD>"] = 0
            self.word2idx["<OOV>"] = 1

        sequences = []
        for text in texts:
            words = str(text).lower().split()
            seq = [self.word2idx.get(w, 1) for w in words]
            if len(seq) < self.max_length:
                seq = seq + [0] * (self.max_length - len(seq))
            else:
                seq = seq[:self.max_length]
            sequences.append(seq)
        return np.array(sequences, dtype=np.int64)

    def fit(self, X, y):
        X_seq = self._tokenize_and_pad(X, fit=True)
        X_tensor = torch.tensor(X_seq, dtype=torch.long).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        actual_vocab_size = min(len(self.word2idx), self.vocab_size)
        self.model = PyTorchBiLSTMClassifier(
            vocab_size=actual_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=2
        ).to(self.device)

        class_counts = np.bincount(y)
        total_samples = len(y)
        weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        X_seq = self._tokenize_and_pad(X, fit=False)
        X_tensor = torch.tensor(X_seq, dtype=torch.long).to(self.device)
        
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch_x = X_tensor[i:i+self.batch_size]
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def export_to_onnx(self, filepath: str):
        """Exporta o modelo BiLSTM treinado para o formato ONNX otimizado para inferência."""
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado antes de ser exportado para ONNX.")
        
        try:
            self.model.eval()
            self.model.to("cpu")
            
            dummy_input = torch.randint(0, self.vocab_size, (1, self.max_length), dtype=torch.long)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                filepath,
                export_params=True,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size'}
                }
            )
            print(f"[SUCESSO] Modelo BiLSTM exportado para ONNX em: {filepath}")
        except Exception as e:
            print(f"[AVISO] Falha ao exportar BiLSTM para ONNX (verifique se 'onnx' e 'onnxscript' estão instalados): {e}")
        finally:
            if self.model is not None:
                self.model.to(self.device)

if __name__ == "__main__":
    import os
    import pandas as pd

    data_path = os.path.join("pre_process_and_analisis", "data", "reviews_ecommerce_aumentados_traducao.csv")
    
    print(f"[INFO] Carregando dataset de treino: {data_path}")
    if not os.path.exists(data_path):
        print(f"[ERRO] Dataset não encontrado em '{data_path}'. Certifique-se de executar o pipeline de tradução primeiro.")
    else:
        df = pd.read_csv(data_path)
        X = df['review_text_processed'].fillna("").astype(str).values
        y = df['polarity'].astype(int).values

        print("[INFO] Treinando o modelo BiLSTM campeão com dados de Back-Translation...")
        bilstm_wrapper = PyTorchBiLSTMWrapper(
            vocab_size=10000,
            max_length=100,
            embedding_dim=100,
            hidden_dim=64,
            max_epochs=5,
            batch_size=64,
            random_state=42
        )
        bilstm_wrapper.fit(X, y)

        onnx_dir = os.path.join("evaluation", "results")
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, "bilstm_model.onnx")
        
        print(f"[INFO] Exportando modelo para ONNX em: {onnx_path}")
        bilstm_wrapper.export_to_onnx(onnx_path)
        print("[SUCESSO] Treinamento e exportação concluídos com sucesso!")