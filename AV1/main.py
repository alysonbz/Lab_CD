import os
import pandas as pd
from pre_process_and_analisis.analise_and_pre_process import run_pipeline
from classification.model_trainer import ModelTrainer
from evaluation.model_evaluator import ModelEvaluator


def main():
    """Arquivo central da aplicação responsável por orquestrar todo o pipeline NLP."""
    DATASET_PATH = 'buscape.csv'
    
    print("=== INICIANDO PIPELINE DO PROJETO DE NLP (BUSCAPÉ) ===")
    
    # 1. Análise Exploratória, Pré-processamento, K-Means/LDA e Aumento de Dados (SMOTE & EDA)
    df_proc, df_smote, df_eda = run_pipeline(DATASET_PATH)
    
    print("\n=== PRÉ-PROCESSAMENTO E AUMENTO DE DADOS CONCLUÍDOS COM SUCESSO ===")

    # 2. Treinamento dos Modelos e Geração das Curvas de Aprendizado (SMOTE vs EDA)
    trainer = ModelTrainer(random_state=42, test_size=0.2, output_dir="evaluation/results")
    experiments_results = {}

    print("\n--- [EXPERIMENTO 1/2] TREINANDO COM DADOS AUMENTADOS VIA SMOTE ---")
    experiments_results["SMOTE"] = trainer.train_models(
        df=df_smote,
        target_column='polarity',
        tipo_dado_column='tipo_dado',
        text_column='review_text_processed',
        epochs=30
    )

    print("\n--- [EXPERIMENTO 2/2] TREINANDO COM DADOS AUMENTADOS VIA EDA ---")
    experiments_results["EDA"] = trainer.train_models(
        df=df_eda,
        target_column='polarity',
        tipo_dado_column='tipo_dado',
        text_column='review_text_processed',
        epochs=30
    )

    print("\n=== TREINAMENTO DE TODOS OS EXPERIMENTOS CONCLUÍDO ===")

    # 3. Avaliação Consolidada dos Resultados
    print("\n=== INICIANDO AVALIAÇÃO CONSOLIDADA DOS MODELOS ===")

    evaluator = ModelEvaluator(output_dir="evaluation/results")
    metrics_df = evaluator.evaluate_models(experiments_results)

    if not metrics_df.empty:
        best_model = metrics_df.iloc[0]

        print("\n" + "*"*60)
        print("=== MELHOR MODELO GLOBAL PELO F1-SCORE ===")
        print(f"Técnica de Aumento: {best_model['Técnica de Aumento']}")
        print(f"Representação:      {best_model['Representação']}")
        print(f"Modelo:             {best_model['Modelo']}")
        print(f"F1-score:           {best_model['F1-score']:.4f}")
        print(f"Acurácia:           {best_model['Acurácia']:.4f}")
        print("*"*60)

    print("\n=== PIPELINE FINALIZADO COM SUCESSO ===")


if __name__ == "__main__":
    main()