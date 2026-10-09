from pre_process_and_analisis.analise_and_pre_process import run_pipeline
from classification.model_trainer import ModelTrainer
from evaluation.model_evaluator import ModelEvaluator


def main():
    """Executa todas as etapas do projeto de NLP."""

    dataset_path = "buscape.csv"

    print("=== INICIANDO PIPELINE DO PROJETO DE NLP (BUSCAPÉ) ===")

    # Partes 1 e 2: pré-processamento e análise exploratória
    df, text_col, label_col = run_pipeline(dataset_path)

    print("\n=== ETAPAS 1 E 2 EXECUTADAS COM SUCESSO ===")

    # Parte 3: treinamento dos classificadores
    print("\n=== INICIANDO CLASSIFICAÇÃO TEXTUAL ===")

    trainer = ModelTrainer()

    results = trainer.train_models(
        texts=df["cleaned_text"],
        labels=df[label_col]
    )

    print("\n=== TREINAMENTO CONCLUÍDO ===")

    # Parte 4: avaliação dos modelos
    print("\n=== INICIANDO AVALIAÇÃO DOS MODELOS ===")

    evaluator = ModelEvaluator()
    metrics_df = evaluator.evaluate_models(results)

    if not metrics_df.empty:
        best_model = metrics_df.iloc[0]

        print("\n=== MELHOR RESULTADO PELO F1-SCORE ===")
        print(f"Representação: {best_model['Representação']}")
        print(f"Modelo: {best_model['Modelo']}")
        print(f"F1-score: {best_model['F1-score']:.4f}")

    print("\n=== PROJETO FINALIZADO ===")


if __name__ == "__main__":
    main()
```
