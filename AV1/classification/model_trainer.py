from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE

from .text_classifier import TextClassifier


class ModelTrainer:
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.classifier = TextClassifier(random_state=random_state)

    def train_models(self, texts, labels):
        """Separa os dados, transforma os textos e treina os classificadores."""

        texts = texts.fillna("").astype(str)
        labels = labels.reset_index(drop=True)
        texts = texts.reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )

        vectorizers = {
            "CountVectorizer": CountVectorizer(max_features=5000),
            "TF-IDF": TfidfVectorizer(max_features=5000)
        }

        models = self.classifier.get_models()

        results = {
            "y_test": y_test,
            "predictions": {},
            "models": {}
        }

        for vectorizer_name, vectorizer in vectorizers.items():
            print(f"\n=== Representação: {vectorizer_name} ===")

            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)

            # O balanceamento é aplicado somente aos dados de treinamento.
            class_counts = y_train.value_counts()

            if len(class_counts) > 1 and class_counts.min() >= 2:
                k_neighbors = min(5, int(class_counts.min()) - 1)

                smote = SMOTE(
                    random_state=self.random_state,
                    k_neighbors=k_neighbors
                )

                X_train_balanced, y_train_balanced = smote.fit_resample(
                    X_train_vectorized,
                    y_train
                )
            else:
                print(
                    "Aviso: não há exemplos suficientes em todas as classes "
                    "para aplicar SMOTE. O treinamento seguirá sem balanceamento."
                )

                X_train_balanced = X_train_vectorized
                y_train_balanced = y_train

            results["predictions"][vectorizer_name] = {}
            results["models"][vectorizer_name] = {}

            for model_name, model in models.items():
                print(f"Treinando modelo: {model_name}")

                model.fit(X_train_balanced, y_train_balanced)

                predictions = model.predict(X_test_vectorized)

                results["predictions"][vectorizer_name][model_name] = (
                    predictions
                )

                results["models"][vectorizer_name][model_name] = model

        results["y_test"] = y_test

        return results
