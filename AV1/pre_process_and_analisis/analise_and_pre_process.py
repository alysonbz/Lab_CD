import os
import pandas as pd
from pre_process_and_analisis.corpus_analyzer import CorpusExploratoryAnalyzer
from pre_process_and_analisis.text_preprocessor import TextPreprocessor


def load_dataset(filepath: str = 'buscape.csv') -> pd.DataFrame:
    """Carrega o dataset estritamente do caminho informado, levantando erro caso não exista."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERRO CRÍTICO] O arquivo do dataset '{filepath}' não foi encontrado."
            ' Encerrando execução.'
        )

    print(f"[INFO] Carregando arquivo '{filepath}'...")
    try:
        return pd.read_csv(filepath)
    except Exception:
        return pd.read_csv(filepath, sep=';')


def run_pipeline(filepath: str = 'buscape.csv'):
    """Executa o pipeline completo integrando SMOTE, EDA e Back-Translation local."""
    df = load_dataset(filepath)

    text_col = 'review_text' if 'review_text' in df.columns else df.columns[0]
    
    if 'polarity' in df.columns:
        label_col = 'polarity'
    elif 'rating' in df.columns:
        label_col = 'rating'
    else:
        label_col = df.columns[-1]

    print(
        f"[INFO] Coluna de texto: '{text_col}' | Coluna de rótulo: '{label_col}'"
    )

    # 1. Análise Exploratória e Métricas Iniciais do Dataset Bruto
    analyzer = CorpusExploratoryAnalyzer(
        df, text_column=text_col, label_column=label_col
    )
    analyzer.get_basic_metrics(save_csv=True)
    analyzer.plot_class_distribution()

    # 2. Execução do Pré-processamento e Aumento de Dados (Incluindo Back-Translation)
    preprocessor = TextPreprocessor(
        language='portuguese',
        enable_spell_check=True,
        n_jobs=-1,
        cache_dir='pre_process_and_analisis/cache_checkpoint'
    )

    df_proc, df_smote, df_eda, df_translation = preprocessor.export_pipeline_results(
        df_original=df,
        text_column=text_col,
        target_column=label_col,
        output_prefix='reviews_ecommerce'
    )

    # 3. Análise Não Supervisionada (K-Means e LDA) no Dataset Original Pré-processado
    if 'review_text_cleaned' in df_proc.columns:
        print("\n[INFO] Aplicando K-Means e LDA no dataset pré-processado...")
        analyzer.apply_unsupervised_analysis(
            df=df_proc,
            text_column='review_text_cleaned',
            n_clusters=4
        )

    print("\n" + "="*60)
    print("[INFO] GERANDO NUVENS DE PALAVRAS (SEM STEMMING)")
    print("="*60)

    # 4. Nuvens de Palavras (Original, EDA, SMOTE, Back-Translation)
    if 'review_text_cleaned' in df_proc.columns:
        print("[INFO] Gerando Nuvens de Palavras: Dataset Original...")
        analyzer_proc = CorpusExploratoryAnalyzer(df_proc, text_column='review_text_cleaned', label_column=label_col)
        analyzer_proc.generate_wordclouds(df_proc, text_column='review_text_cleaned', label_column=label_col)

    if 'review_text_cleaned' in df_eda.columns:
        print("[INFO] Gerando Nuvens de Palavras: Dataset Aumentado EDA...")
        analyzer_eda = CorpusExploratoryAnalyzer(df_eda, text_column='review_text_cleaned', label_column=label_col)
        analyzer_eda.generate_wordclouds(df_eda, text_column='review_text_cleaned', label_column=label_col)

    if df_translation is not None and 'review_text_cleaned' in df_translation.columns:
        print("[INFO] Gerando Nuvens de Palavras: Dataset Aumentado por Tradução...")
        analyzer_trans = CorpusExploratoryAnalyzer(df_translation, text_column='review_text_cleaned', label_column=label_col)
        analyzer_trans.generate_wordclouds(df_translation, text_column='review_text_cleaned', label_column=label_col)

    if 'review_text_cleaned' in df_smote.columns:
        print("[INFO] Gerando Nuvens de Palavras: Dataset Aumentado SMOTE...")
        analyzer_smote = CorpusExploratoryAnalyzer(df_smote, text_column='review_text_cleaned', label_column=label_col)
        analyzer_smote.generate_wordclouds(df_smote, text_column='review_text_cleaned', label_column=label_col)
    else:
        tfidf_cols = [c for c in df_smote.columns if c.startswith('tfidf_')]
        if tfidf_cols:
            print("[INFO] Gerando Nuvem de Palavras: Dataset Aumentado SMOTE (reconstruído via TF-IDF)...")
            word_sums = df_smote[tfidf_cols].sum(axis=0)
            word_sums.index = word_sums.index.str.replace('tfidf_', '', regex=False)
            
            max_val = word_sums.max()
            if max_val > 0:
                word_counts = (word_sums / max_val * 100).astype(int)
                smote_text = " ".join([f"{word} " * count for word, count in word_counts.items() if count > 0])
                smote_series = pd.Series([smote_text])
                
                analyzer_smote = CorpusExploratoryAnalyzer(df_smote, text_column=label_col, label_column=label_col)
                analyzer_smote.generate_wordclouds(pd.DataFrame({'review_text_cleaned': smote_series}), text_column='review_text_cleaned', label_column=label_col)

    print(
        '\n[SUCESSO] Pipeline de Análise, Pré-processamento e Nuvens de Palavras concluído.'
    )
    return df_proc, df_smote, df_eda, df_translation


if __name__ == '__main__':
    DATASET_PATH = 'buscape.csv'
    run_pipeline(DATASET_PATH)