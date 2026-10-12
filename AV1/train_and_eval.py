import os
import pandas as pd
from classification.model_trainer import ModelTrainer
from evaluation.model_evaluator import ModelEvaluator


def main():
    """Script otimizado para carregar os datasets do disco e executar apenas o treinamento e a avaliação."""
    data_dir = "pre_process_and_analisis/data"
    
    path_smote = os.path.join(data_dir, "reviews_ecommerce_aumentados_smote.csv")
    path_eda = os.path.join(data_dir, "reviews_ecommerce_aumentados_eda.csv")
    path_translation = os.path.join(data_dir, "reviews_ecommerce_aumentados_traducao.csv")

    print("=== CARREGANDO DATASETS JÁ PROCESSADOS DO DISCO ===")
    
    if not (os.path.exists(path_smote) and os.path.exists(path_eda) and os.path.exists(path_translation)):
        raise FileNotFoundError(
            "[ERRO] Um ou mais arquivos de dataset não foram encontrados em 'pre_process_and_analisis/data/'. "
            "Certifique-se de ter executado o pré-processamento ao menos uma vez."
        )

    df_smote = pd.read_csv(path_smote)
    df_eda = pd.read_csv(path_eda)
    df_translation = pd.read_csv(path_translation)
    
    print("[SUCESSO] Datasets carregados com sucesso:")
    print(f" 📦 SMOTE: {len(df_smote):,} linhas")
    print(f" 📦 EDA: {len(df_eda):,} linhas")
    print(f" 📦 Tradução: {len(df_translation):,} linhas\n")

    trainer = ModelTrainer(random_state=42, test_size=0.2, output_dir="evaluation/results")
    experiments_results = {}

    print("--- [EXPERIMENTO 1/3] TREINANDO COM DADOS AUMENTADOS VIA SMOTE ---")
    experiments_results["SMOTE"] = trainer.train_models(
        df=df_smote,
        target_column='polarity',
        tipo_dado_column='tipo_dado',
        text_column='review_text_processed',
        epochs=30
    )

    print("\n--- [EXPERIMENTO 2/3] TREINANDO COM DADOS AUMENTADOS VIA EDA ---")
    experiments_results["EDA"] = trainer.train_models(
        df=df_eda,
        target_column='polarity',
        tipo_dado_column='tipo_dado',
        text_column='review_text_processed',
        epochs=30
    )

    print("\n--- [EXPERIMENTO 3/3] TREINANDO COM DADOS AUMENTADOS VIA TRADUÇÃO ---")
    experiments_results["Tradução"] = trainer.train_models(
        df=df_translation,
        target_column='polarity',
        tipo_dado_column='tipo_dado',
        text_column='review_text_processed',
        epochs=30
    )

    print("\n=== TREINAMENTO DE TODOS OS EXPERIMENTOS CONCLUÍDO ===")

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

    print("\n=== PIPELINE DE TREINAMENTO E AVALIAÇÃO FINALIZADO COM SUCESSO ===")


if __name__ == "__main__":
    main()