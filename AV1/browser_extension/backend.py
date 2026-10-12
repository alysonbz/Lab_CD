import sys
import os

# Adiciona a raiz do projeto (AV1) ao sys.path para conseguir importar de 'classification'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from flask import Flask, request, jsonify
from flask_cors import CORS
from classification.bilstm_translation_augmented_inference import SentimentInference

app = Flask(__name__)
CORS(app)

print("[INFO] Inicializando motor de inferência ONNX para a API...")
try:
    # Define explicitamente os caminhos absolutos a partir da raiz do projeto (AV1)
    dataset_path = os.path.join(parent_dir, "pre_process_and_analisis", "data", "reviews_ecommerce_aumentados_traducao.csv")
    onnx_path = os.path.join(parent_dir, "evaluation", "results", "bilstm_model.onnx")
    
    # Inicializa o motor de inferência passando os caminhos corretos
    inference_engine = SentimentInference(dataset_path=dataset_path, onnx_path=onnx_path)
except Exception as e:
    print(f"[ERRO CRÍTICO] Falha ao carregar o modelo de inferência: {e}")
    inference_engine = None


@app.route('/predict', methods=['POST'])
def predict():
    if inference_engine is None:
        return jsonify({'error': 'Modelo de inferência não foi carregado corretamente no servidor.'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Campo "text" não fornecido no corpo da requisição.'}), 400

    text = data.get('text', '')
    print(f"[INFO] Recebido texto para análise: {text}")

    try:
        # Executa a predição utilizando a BiLSTM via ONNX Runtime
        result = inference_engine.predict(text)
        
        # Mapeia a saída para o formato esperado pelo frontend
        sentiment_label = 'positivo' if result['prediction'] == 1 else 'negativo'
        
        response_data = {
            'text': result['text'],
            'sentiment': sentiment_label,
            'label_name': result['label_name'],
            'confidence': round(result['confidence'], 2)
        }
        
        print(f"[INFO] Sentimento previsto: {sentiment_label} ({result['confidence']:.2f}%)")
        return jsonify(response_data)

    except Exception as e:
        print(f"[ERRO] Falha durante a inferência: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("[INFO] Servidor Flask rodando na porta 5000...")
    app.run(port=5000, debug=False)