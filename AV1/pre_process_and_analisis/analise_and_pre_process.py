
import os
import pandas as pd
from pre_process_and_analisis.corpus_analyzer import CorpusExploratoryAnalyzer
from pre_process_and_analisis.text_preprocessor import TextPreprocessor


def load_dataset(filepath: str = 'buscape.csv') -> pd.DataFrame:
    """Carrega o dataset do caminho informado."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERRO CRÍTICO] O arquivo do dataset '{filepath}' "
            "não foi encontrado. Encerrando execução."
        )

    print(f"[INFO] Carregando arquivo '{filepath}'...")

    try:
        return pd.read_csv(filepath)
    except Exception:
        return pd.read_csv(filepath, sep=';')


def run_pipeline(filepath: str = 'buscape.csv'):
    """Executa as etapas de pré-processamento e análise exploratória."""

    # Carregamento do dataset
    df = load_dataset(filepath)

    # Identificação das colunas de texto e classificação
    text_col = (
        'review_text'
        if 'review_text' in df.columns
        else df.columns[0]
    )

    label_col = (
        'rating'
        if 'rating' in df.columns
        else df.columns[-1]
    )

    print(
        f"[INFO] Coluna de texto: '{text_col}' | "
        f"Coluna de rótulo: '{label_col}'"
    )

    # Parte 2: análise exploratória do corpus
    analyzer = CorpusExploratoryAnalyzer(
        df,
        text_column=text_col,
        label_column=label_col
    )

    analyzer.get_basic_metrics(save_csv=True)
    analyzer.plot_class_distribution()
    analyzer.suggest_unsupervised_applications()

    # Parte 1: pré-processamento e limpeza dos textos
    preprocessor = TextPreprocessor(language='portuguese')

    df['cleaned_text'] = preprocessor.preprocess_corpus(
        df[text_col]
    )

    analyzer.generate_wordcloud(df['cleaned_text'])

    print(
        "[SUCESSO] Etapas de pré-processamento e "
        "análise exploratória concluídas."
    )

    # Retorna os textos limpos e as informações das colunas.
    # A vetorização e o balanceamento serão realizados depois, durante o treinamento dos modelos.
    return df, text_col, label_col


if __name__ == '__main__':
    DATASET_PATH = 'buscape.csv'

    run_pipeline(DATASET_PATH)
