from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


class ModelEvaluator:
    def __init__(self, output_dir="evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_models(self, experiments_results: dict) -> pd.DataFrame:
        """
        Calcula as métricas e gera relatórios e matrizes de confusão para
        múltiplos experimentos (ex: SMOTE vs EDA e CountVectorizer vs TF-IDF).
        
        Esperado em `experiments_results`:
        {
            "SMOTE": {
                "y_test": array,
                "predictions": {
                    "CountVectorizer": {"ModeloA": y_pred, ...},
                    "TF-IDF": {"ModeloA": y_pred, ...}
                }
            },
            "EDA": { ... }
        }
        """
        metrics = []

        # Caso seja passado um único dicionário tradicional de resultados, empacota com chave padrão
        if "y_test" in experiments_results:
            experiments_results = {"Geral": experiments_results}

        for technique_name, results in experiments_results.items():
            y_test = results["y_test"]
            predictions = results["predictions"]

            for vectorizer_name, model_predictions in predictions.items():
                for model_name, y_pred in model_predictions.items():

                    accuracy = accuracy_score(y_test, y_pred)

                    precision = precision_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0
                    )

                    recall = recall_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0
                    )

                    metrics.append({
                        "Técnica de Aumento": technique_name,
                        "Representação": vectorizer_name,
                        "Modelo": model_name,
                        "Acurácia": accuracy,
                        "Precisão": precision,
                        "Recall": recall,
                        "F1-score": f1
                    })

                    print(
                        f"\n=== [{technique_name}] {model_name} com {vectorizer_name} ==="
                    )

                    print(
                        classification_report(
                            y_test,
                            y_pred,
                            zero_division=0
                        )
                    )

                    self._save_confusion_matrix(
                        y_test,
                        y_pred,
                        model_name=model_name,
                        vectorizer_name=vectorizer_name,
                        technique_name=technique_name
                    )

        metrics_df = pd.DataFrame(metrics)

        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values(
                by="F1-score",
                ascending=False
            ).reset_index(drop=True)

            metrics_path = self.output_dir / "metrics_comparison.csv"
            metrics_df.to_csv(
                metrics_path,
                index=False,
                encoding="utf-8-sig"
            )

            print("\n" + "="*70)
            print("=== COMPARAÇÃO CONSOLIDADA DOS MODELOS (SMOTE vs EDA | Word Vec vs TF-IDF) ===")
            print("="*70)
            print(metrics_df.to_string(index=False))

            print(f"\nMétricas salvas em: {metrics_path}")

        return metrics_df

    def _save_confusion_matrix(
        self,
        y_test,
        y_pred,
        model_name: str,
        vectorizer_name: str,
        technique_name: str
    ):
        """Gera e salva a matriz de confusão identificando a técnica e representação."""

        figure, axis = plt.subplots(figsize=(8, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=axis,
            xticks_rotation="vertical",
            colorbar=False
        )

        axis.set_title(
            f"Matriz de Confusão - {model_name}\n({technique_name} | {vectorizer_name})"
        )

        figure.tight_layout()

        filename = (
            f"confusion_matrix_{technique_name}_{model_name}_{vectorizer_name}"
            .replace(" ", "_")
            .replace("/", "_")
            + ".png"
        )

        figure.savefig(
            self.output_dir / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(figure)