import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import onnxruntime as ort
from collections import Counter

from pre_process_and_analisis.text_cleaner import worker_clean_chunk
from pre_process_and_analisis.lemmatizer import worker_lemmatize_chunk


class SentimentInference:
    """Classe de inferência integrada ao pipeline oficial de pré-processamento e modelo ONNX."""
    def __init__(self, dataset_path: str = None, onnx_path: str = None, max_length: int = 100, vocab_size: int = 10000):
        if dataset_path is None:
            dataset_path = os.path.join("pre_process_and_analisis", "data", "reviews_ecommerce_aumentados_traducao.csv")
        if onnx_path is None:
            onnx_path = os.path.join("evaluation", "results", "bilstm_model.onnx")

        print("[INFO] Carregando dataset para reconstruir o vocabulário...")
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"[ERRO] O arquivo '{dataset_path}' não foi encontrado. Verifique o caminho.")

        X_train = df['review_text_processed'].fillna("").astype(str).values
        word_counts = Counter()
        for text in X_train:
            words = str(text).lower().split()
            word_counts.update(words)
        
        most_common = word_counts.most_common(vocab_size - 2)
        self.word2idx = {word: idx + 2 for idx, (word, count) in enumerate(most_common)}
        self.word2idx["<PAD>"] = 0
        self.word2idx["<OOV>"] = 1
        self.max_length = max_length

        print(f"[INFO] Carregando motor de inferência ONNX Runtime: {onnx_path}")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"[ERRO] Arquivo ONNX não encontrado em '{onnx_path}'. Execute o treinamento do bilstm.py primeiro.")
        
        self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("[SUCESSO] Sistema de inferência pronto para uso!\n")

    def _preprocess_text(self, texts):
        """Aplica o mesmo pipeline de limpeza e lematização do treinamento nos novos textos."""
        if isinstance(texts, str):
            texts = [texts]
        
        cleaned = worker_clean_chunk(list(texts))
        
        processed = worker_lemmatize_chunk((cleaned, 'portuguese'))
        
        return processed

    def _tokenize_and_pad(self, texts):
        processed_texts = self._preprocess_text(texts)
        sequences = []
        for text in processed_texts:
            words = str(text).lower().split()
            seq = [self.word2idx.get(w, 1) for w in words]
            if len(seq) < self.max_length:
                seq = seq + [0] * (self.max_length - len(seq))
            else:
                seq = seq[:self.max_length]
            sequences.append(seq)
        return np.array(sequences, dtype=np.int64)

    def predict(self, texts):
        """Recebe texto(s) bruto(s), pré-processa e retorna a predição via ONNX."""
        is_single = False
        if isinstance(texts, str):
            texts = [texts]
            is_single = True

        input_seq = self._tokenize_and_pad(texts)
        
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_seq}
        logits = self.ort_session.run(None, ort_inputs)[0]
        
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        predictions = np.argmax(probabilities, axis=1)

        results = []
        for text, pred, prob in zip(texts, predictions, probabilities):
            label_name = "Positivo (1)" if pred == 1 else "Negativo (0)"
            confidence = float(prob[pred]) * 100
            results.append({
                "text": text,
                "prediction": int(pred),
                "label_name": label_name,
                "confidence": confidence
            })

        return results[0] if is_single else results


def main():
    try:
        inference_engine = SentimentInference()
    except Exception as e:
        print(e)
        return

    print("=" * 60)
    print("      SISTEMA DE INFERÊNCIA ONNX - ANÁLISE DE SENTIMENTOS")
    print("=" * 60)
    print("Escolha o modo de operação:")
    print(" [1] Executar frases de teste pré-definidas")
    print(" [2] Modo interativo (Digitar seus próprios reviews)")
    print("-" * 60)

    modo = input("Digite o número da opção desejada (1 ou 2): ").strip()

    if modo == "1":
        print("\n--- MODO 1: FRASES DE TESTE PRÉ-DEFINIDAS ---")
        test_phrases = [
            "O smartphone é excelente, a câmera tira fotos maravilhosas e a bateria dura bastante.",
            "Péssimo produto! Veio com defeito de fábrica, travando o tempo todo e o suporte não resolveu.",
            "Atendeu minhas expectativas pelo preço cobrado, entrega rápida e bem embalado.",
            "Não gostei do material, plástico muito fraco e quebrou na primeira semana de uso."
        ]

        results = inference_engine.predict(test_phrases)
        for res in results:
            print("-" * 50)
            print(f"Review: {res['text']}")
            print(f"Sentimento Previsto: {res['label_name']}")
            print(f"Confiança: {res['confidence']:.2f}%")
        print("-" * 50)

    elif modo == "2":
        print("\n--- MODO 2: ENTRADA INTERATIVA DO USUÁRIO ---")
        print("Dica: Digite 'sair', 'exit' ou 'quit' a qualquer momento para encerrar.\n")

        while True:
            texto_usuario = input("Digite o review do produto: ").strip()
            
            if texto_usuario.lower() in ["sair", "exit", "quit"]:
                print("\nEncerrando o modo interativo. Até logo!")
                break
            
            if not texto_usuario:
                print("[AVISO] Por favor, digite um texto válido.\n")
                continue

            res = inference_engine.predict(texto_usuario)
            print("-" * 50)
            print(f"Sentimento Previsto: {res['label_name']}")
            print(f"Confiança: {res['confidence']:.2f}%")
            print("-" * 50 + "\n")
            
    else:
        print("\n[ERRO] Opção inválida! Execute o script novamente e digite '1' ou '2'.")


if __name__ == "__main__":
    main()