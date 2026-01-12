import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def executar_classificacao():
    print("Carregando atributos dos CSVs...")

    # 1. Leitura dos arquivos gerados no tópico anterior
    df_train = pd.read_csv('atributos_train.csv')
    df_test = pd.read_csv('atributos_test.csv')

    # 2. Separação de X (atributos) e y (rótulo)
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']

    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    print(f"Treinando o modelo com {len(X_train)} amostras")

    # 3. Classificação (Tópico 5)
    # Usamos Random Forest por ser robusto a outliers e capturar relações não lineares
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # 4. Predição
    y_pred = modelo.predict(X_test)

    # 5. Geração de métricas (Tópico 6)
    acuracia = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 30)
    print("=" * 30)
    print(f"Acurácia Global: {acuracia:.2%}")

    print("\nRelatório de Classificação:")
    # 0 = Normal, 1 = Pneumonia
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia']))

    # Plot da Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.title('Matriz de Confusão - Classificação de Raio-X')
    plt.savefig('matriz_confusao.png')  # Salva a imagem para usar no relatório
    print("\nGráfico 'matriz_confusao.png' salvo com sucesso.")
    plt.show()


if __name__ == "__main__":
    executar_classificacao()