from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    # Mock simulado para teste:
    print(f"[INFO] Recebido texto para análise: {text}")
    sentiment = 'negativo' if 'horrível' in text.lower() or 'péssimo' in text.lower() else 'positivo'
    print(f"[INFO] Sentimento previsto: {sentiment}")
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(port=5000)