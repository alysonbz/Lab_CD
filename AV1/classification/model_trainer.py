from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from .text_classifier import TextClassifier


class ModelTrainer:
    def __init__(self, random_state: int = 42, test_size: float = 0.2, output_dir: str = "evaluation/results"):
        self.random_state = random_state
        self.test_size = test_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = TextClassifier(random_state=random_state)

    def train_models(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'polarity', 
        tipo_dado_column: str = 'tipo_dado',
        text_column: str = 'review_text_processed',
        epochs: int = 30
    ):
        df = df.copy()

        # 1. Separação de dados originais e sintéticos
        df_original = df[df[tipo_dado_column] == 'original'].reset_index(drop=True)
        df_synthetic = df[df[tipo_dado_column] != 'original'].reset_index(drop=True)

        print(f"[INFO] Registros originais: {len(df_original)}")
        print(f"[INFO] Registros sintéticos/aumentados: {len(df_synthetic)}")

        # 2. Divisão Treino/Teste apenas com os dados originais
        df_train_orig, df_test = train_test_split(
            df_original,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df_original[target_column]
        )

        # 3. Unificação do Treino (Originais de Treino + Sintéticos)
        df_train = pd.concat([df_train_orig, df_synthetic], axis=0)
        df_train = df_train.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        print(f"[INFO] Treino Final: {len(df_train)} amostras ({len(df_train_orig)} originais + {len(df_synthetic)} sintéticas)")
        print(f"[INFO] Teste Final: {len(df_test)} amostras (100% originais)\n")

        y_train = df_train[target_column].astype(int).values
        y_test = df_test[target_column].astype(int).values
        classes = np.unique(y_train)

        results = {
            "y_test": y_test,
            "predictions": {},
            "models": {}
        }

        # 4. Verificação de dataset pré-vetorizado vs textual
        feature_cols = [col for col in df.columns if col not in [target_column, tipo_dado_column, text_column, 'review_text_cleaned', 'review_text']]
        is_pre_vectorized = len(feature_cols) > 0 and all(np.issubdtype(df[col].dtype, np.number) for col in feature_cols)

        if is_pre_vectorized:
            representations = {"PreVectorized": (df_train[feature_cols].values, df_test[feature_cols].values)}
        else:
            X_train_text = df_train[text_column].fillna("").astype(str)
            X_test_text = df_test[text_column].fillna("").astype(str)

            vectorizers = {
                "CountVectorizer": CountVectorizer(max_features=5000),
                "TF-IDF": TfidfVectorizer(max_features=5000)
            }

            representations = {}
            for name, vec in vectorizers.items():
                representations[name] = (vec.fit_transform(X_train_text), vec.transform(X_test_text))

        # 5. Treinamento dos 4 modelos originais
        for rep_name, (X_train, X_test) in representations.items():
            print(f"\n=== Representação: {rep_name} ===")

            results["predictions"][rep_name] = {}
            results["models"][rep_name] = {}

            models = self.classifier.get_models()

            for model_name, model in models.items():
                print(f"Treinando e analisando aprendizado: {model_name}")

                # --- MODELOS ITERATIVOS (MLP e Regressão Logística) ---
                if isinstance(model, (MLPClassifier, LogisticRegression)):
                    history = self._train_epoch_model(model, X_train, y_train, X_test, y_test, classes, epochs)
                    self._plot_epoch_curves(history, model_name, rep_name, epochs)
                    
                    # Predição final
                    y_pred = model.predict(X_test)

                # --- MODELOS NÃO ITERATIVOS (Naive Bayes e SVM) ---
                else:
                    # Treinamento direto no conjunto completo
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Análise de aprendizado variando o tamanho das amostras de treino
                    history = self._evaluate_learning_by_sample_size(model, X_train, y_train, X_test, y_test, classes)
                    self._plot_sample_size_curves(history, model_name, rep_name)

                results["predictions"][rep_name][model_name] = y_pred
                results["models"][rep_name][model_name] = model

        return results

    def _train_epoch_model(self, model, X_train, y_train, X_test, y_test, classes, epochs):
        """Treina modelos iterativos acumulando loss e acurácia por época."""
        history = {"loss_train": [], "accuracy_train": [], "loss_test": [], "accuracy_test": []}

        if hasattr(model, 'warm_start'):
            model.warm_start = True
            if hasattr(model, 'max_iter'):
                 model.max_iter = 100 
            if isinstance(model, MLPClassifier):
                 model.tol = 1e-4 

        for epoch in range(1, epochs + 1):
            try:
                model.fit(X_train, y_train)
            except Exception:
                pass 

            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)

            acc_tr = accuracy_score(y_train, y_tr_pred)
            acc_te = accuracy_score(y_test, y_te_pred)

            if hasattr(model, "predict_proba"):
                y_tr_prob = model.predict_proba(X_train)
                y_te_prob = model.predict_proba(X_test)
                l_tr = log_loss(y_train, y_tr_prob, labels=classes)
                l_te = log_loss(y_test, y_te_prob, labels=classes)
            else:
                l_tr = zero_one_loss(y_train, y_tr_pred)
                l_te = zero_one_loss(y_test, y_te_pred)

            history["loss_train"].append(l_tr)
            history["accuracy_train"].append(acc_tr)
            history["loss_test"].append(l_te)
            history["accuracy_test"].append(acc_te)

        return history

    def _evaluate_learning_by_sample_size(self, model_class, X_train, y_train, X_test, y_test, classes, steps=10):
        """Mede a evolução do aprendizado para modelos não iterativos (ex: Naive Bayes, SVM) variando a fração de dados."""
        history = {"fractions": [], "loss_train": [], "accuracy_train": [], "loss_test": [], "accuracy_test": []}
        
        fractions = np.linspace(0.1, 1.0, steps)
        total_samples = X_train.shape[0]

        for frac in fractions:
            n_samples = int(total_samples * frac)
            X_sub = X_train[:n_samples]
            y_sub = y_train[:n_samples]

            # Clona/reinstancia o modelo para treinar no subconjunto
            sub_model = model_class.__class__(**model_class.get_params())
            sub_model.fit(X_sub, y_sub)

            y_tr_pred = sub_model.predict(X_sub)
            y_te_pred = sub_model.predict(X_test)

            acc_tr = accuracy_score(y_sub, y_tr_pred)
            acc_te = accuracy_score(y_test, y_te_pred)

            if hasattr(sub_model, "predict_proba"):
                y_tr_prob = sub_model.predict_proba(X_sub)
                y_te_prob = sub_model.predict_proba(X_test)
                l_tr = log_loss(y_sub, y_tr_prob, labels=classes)
                l_te = log_loss(y_test, y_te_prob, labels=classes)
            else:
                l_tr = zero_one_loss(y_sub, y_tr_pred)
                l_te = zero_one_loss(y_test, y_te_pred)

            history["fractions"].append(int(frac * 100))
            history["loss_train"].append(l_tr)
            history["accuracy_train"].append(acc_tr)
            history["loss_test"].append(l_te)
            history["accuracy_test"].append(acc_te)

        return history

    def _plot_epoch_curves(self, history: dict, model_name: str, rep_name: str, epochs: int):
        """Plota curvas de aprendizado por ÉPOCAS (MLP, Regressão Logística)."""
        epochs_range = range(1, epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history["loss_train"], 'b-o', label='Loss Treino', linewidth=2, markersize=4)
        ax1.plot(epochs_range, history["loss_test"], 'r--s', label='Loss Teste', linewidth=2, markersize=4)
        ax1.set_title(f'Perda (Loss) vs Épocas\n({model_name} - {rep_name})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Época', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.plot(epochs_range, history["accuracy_train"], 'b-o', label='Acurácia Treino', linewidth=2, markersize=4)
        ax2.plot(epochs_range, history["accuracy_test"], 'g--s', label='Acurácia Teste', linewidth=2, markersize=4)
        ax2.set_title(f'Acurácia vs Épocas\n({model_name} - {rep_name})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Época', fontsize=10)
        ax2.set_ylabel('Acurácia', fontsize=10)
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        filename = self.output_dir / f"learning_curve_epochs_{model_name}_{rep_name}.png".replace(" ", "_")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_sample_size_curves(self, history: dict, model_name: str, rep_name: str):
        """Plota curvas de aprendizado por TAMANHO DO DATASET (Naive Bayes, SVM)."""
        x_axis = history["fractions"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(x_axis, history["loss_train"], 'b-o', label='Loss/Erro Treino', linewidth=2, markersize=5)
        ax1.plot(x_axis, history["loss_test"], 'r--s', label='Loss/Erro Teste', linewidth=2, markersize=5)
        ax1.set_title(f'Curva de Aprendizado (Perda vs % Amostras)\n({model_name} - {rep_name})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('% de Amostras de Treino Utilizadas', fontsize=10)
        ax1.set_ylabel('Perda / Taxa de Erro', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.plot(x_axis, history["accuracy_train"], 'b-o', label='Acurácia Treino', linewidth=2, markersize=5)
        ax2.plot(x_axis, history["accuracy_test"], 'g--s', label='Acurácia Teste', linewidth=2, markersize=5)
        ax2.set_title(f'Curva de Aprendizado (Acurácia vs % Amostras)\n({model_name} - {rep_name})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('% de Amostras de Treino Utilizadas', fontsize=10)
        ax2.set_ylabel('Acurácia', fontsize=10)
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        filename = self.output_dir / f"learning_curve_samples_{model_name}_{rep_name}.png".replace(" ", "_")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)