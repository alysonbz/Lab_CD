import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# Importando os classificadores
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def executar_grid_search_avancado():
    print("Carregando dados...")
    df_train = pd.read_csv('atributos_train.csv')
    df_test = pd.read_csv('atributos_test.csv')

    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    print("Iniciando Grid Search com múltiplos modelos...")

    # 1. Definindo os Pipelines para cada modelo
    # Usamos MinMaxScaler como padrão pois funcionou bem nos testes anteriores
    pipelines = {
        'LogisticRegression': Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000))
        ]),
        'SVM': Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', SVC(class_weight='balanced', random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('scaler', None),  # RF não precisa de escala, mas o pipeline exige o passo
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ]),
        'KNN': Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', KNeighborsClassifier())
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', None),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
    }

    # 2. Definindo a Grade de Parâmetros para cada modelo
    # Sintaxe: 'nome_do_passo__parametro': [lista]
    param_grids = {
        'LogisticRegression': {
            'clf__C': [0.1, 1, 10, 100],
            'clf__solver': ['liblinear', 'lbfgs']
        },
        'SVM': {
            'clf__C': [1, 10, 100],
            'clf__kernel': ['rbf', 'poly'],
            'clf__gamma': ['scale', 0.1]
        },
        'RandomForest': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5]
        },
        'KNN': {
            'clf__n_neighbors': [3, 5, 7, 9],
            'clf__weights': ['uniform', 'distance']
        },
        'GradientBoosting': {
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.01, 0.1],
            'clf__max_depth': [3, 5]
        }
    }

    # Variáveis para armazenar o campeão
    best_overall_model = None
    best_overall_acc = 0.0
    best_model_name = ""
    results_list = []

    # 3. Loop de Execução
    for name, pipeline in pipelines.items():
        print(f"\n--- Otimizando {name} ---")
        # cv=3 para ser mais rápido, pode aumentar para 5
        grid = GridSearchCV(pipeline, param_grids[name], cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Avaliar no Teste
        y_pred = grid.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"  Melhores Params: {grid.best_params_}")
        print(f"  Acurácia Teste: {test_acc:.2%}")

        results_list.append({'Modelo': name, 'Acurácia': test_acc})

        # Verificar se é o novo campeão
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_overall_model = grid.best_estimator_
            best_model_name = name

    print("\n" + "=" * 40)
    print(f"VENCEDOR: {best_model_name} com {best_overall_acc:.2%}")
    print("=" * 40)

    # Detalhes do Vencedor
    y_pred_final = best_overall_model.predict(X_test)
    print(classification_report(y_test, y_pred_final, target_names=['Normal', 'Pneumonia']))

    # Plot Matriz Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_final), annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Matriz de Confusão: {best_model_name} (Otimizado)')
    plt.savefig('matriz_confusao_campeao.png')
    plt.show()


if __name__ == "__main__":
    executar_grid_search_avancado()