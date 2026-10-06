import os
import pandas as pd
from pre_process_and_analisis.text_preprocessor import TextPreprocessor
from pre_process_and_analisis.corpus_analyzer import CorpusExploratoryAnalyzer

def load_dataset(filepath: str = 'buscape.csv') -> pd.DataFrame:
    """Carrega o dataset estritamente do caminho informado, levantando erro caso não exista."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERRO CRÍTICO] O arquivo do dataset '{filepath}' não foi encontrado. Encerrando execução.")
    
    print(f"[INFO] Carregando arquivo '{filepath}'...")
    try:
        return pd.read_csv(filepath)
    except Exception:
        return pd.read_csv(filepath, sep=';')

def run_pipeline(filepath: str = 'buscape.csv'):
    """Executa o pipeline completo da Parte 1 e Parte 2, retornando os dados processados e vetorizados."""
    # Carregamento estrito (lança exceção se ausente)
    df = load_dataset(filepath)

    # Identificação dinâmica das colunas de texto e rótulo
    text_col = 'review_text' if 'review_text' in df.columns else df.columns[0]
    label_col = 'rating' if 'rating' in df.columns else df.columns[-1]
    print(f"[INFO] Coluna de texto: '{text_col}' | Coluna de rótulo: '{label_col}'")

    # Parte 2: Análise Exploratória
    analyzer = CorpusExploratoryAnalyzer(df, text_column=text_col, label_column=label_col)
    analyzer.get_basic_metrics()
    analyzer.plot_class_distribution()
    analyzer.suggest_unsupervised_applications()

    # Parte 1: Pré-processamento e Limpeza
    preprocessor = TextPreprocessor(language='portuguese')
    df['cleaned_text'] = preprocessor.preprocess_corpus(df[text_col])
    analyzer.generate_wordcloud(df['cleaned_text'])

    # Vetorização (TF-IDF) e balanceamento
    X_tfidf = preprocessor.transform_to_features(df['cleaned_text'], method='tfidf')
    y = df[label_col]
    
    X_resampled, y_resampled = preprocessor.handle_imbalance(X_tfidf, y)

    print(f"[SUCESSO] Pipeline de Análise e Pré-processamento concluído. Features shape: {X_resampled.shape}")
    return X_resampled, y_resampled, df

if __name__ == "__main__":
    # Permite execução independente do script
    DATASET_PATH = 'buscape.csv'
    run_pipeline(DATASET_PATH)