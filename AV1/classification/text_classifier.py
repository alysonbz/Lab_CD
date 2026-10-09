from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class TextClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_models(self):
        """Retorna os 4 modelos de classificação de texto originais do projeto."""
        models = {
            "Naive Bayes": MultinomialNB(),

            "SVM": LinearSVC(
                random_state=self.random_state,
                dual="auto"
            ),

            "Regressão Logística": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),

            "MLP": MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=300,
                random_state=self.random_state
            )
        }

        return models