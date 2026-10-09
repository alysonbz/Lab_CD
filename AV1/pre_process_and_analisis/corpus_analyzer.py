from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


class CorpusExploratoryAnalyzer:
  """Classe focada em análise exploratória de dados (EDA), métricas estatísticas,

  visualizações gráficas, exportação de relatórios CSV e tarefas não
  supervisionadas.
  """

  def __init__(
      self,
      df: pd.DataFrame,
      text_column: str,
      label_column: str,
      output_dir: str = None,
  ):
    self.df = df
    self.text_column = text_column
    self.label_column = label_column

    # Diretório padrão para saídas (visualizations e CSVs)
    if output_dir:
      self.output_dir = Path(output_dir)
    else:
      self.output_dir = Path(__file__).resolve().parent / 'visualizations'

    self.output_dir.mkdir(parents=True, exist_ok=True)

  def get_basic_metrics(self, save_csv: bool = True) -> dict:
    """Calcula métricas descritivas do corpus e exporta os relatórios para CSV."""
    total_docs = len(self.df)
    num_classes = self.df[self.label_column].nunique()
    class_distribution = self.df[self.label_column].value_counts()

    # Análise de texto (caracteres e palavras)
    text_series = self.df[self.text_column].fillna('').astype(str)
    char_lengths = text_series.apply(len)
    word_counts = text_series.apply(lambda x: len(x.split()))
    null_count = self.df[self.text_column].isnull().sum()

    # 1. Tabela Resumo das Métricas
    metrics_summary = pd.DataFrame({
        'metrica': [
            'Tamanho Total do Corpus (Docs)',
            'Número de Classes Únicas',
            'Documentos Nulos/Vazios',
            'Média de Caracteres por Doc',
            'Mínimo de Caracteres por Doc',
            'Máximo de Caracteres por Doc',
            'Média de Palavras por Doc',
            'Mínimo de Palavras por Doc',
            'Máximo de Palavras por Doc',
            'Total Geral de Palavras',
        ],
        'valor': [
            total_docs,
            num_classes,
            null_count,
            round(float(char_lengths.mean()), 2),
            int(char_lengths.min()),
            int(char_lengths.max()),
            round(float(word_counts.mean()), 2),
            int(word_counts.min()),
            int(word_counts.max()),
            int(word_counts.sum()),
        ],
    })

    # 2. Tabela de Distribuição de Classes (Correção do .round no array NumPy)
    df_class_dist = pd.DataFrame({
        'classe': class_distribution.index,
        'quantidade': class_distribution.values,
        'percentual': np.round(
            (class_distribution.values / total_docs) * 100, 2
        ),
    })

    # Exibição no Console
    print('=' * 50)
    print('MÉTRICAS BÁSICAS E DESCRITIVAS DO CORPUS')
    print('=' * 50)
    print(f'• Tamanho total do corpus: {total_docs} documentos')
    print(f'• Número de classes únicas: {num_classes}')
    print(f'• Média de palavras/doc: {round(float(word_counts.mean()), 2)}')
    print(f'• Média de caracteres/doc: {round(float(char_lengths.mean()), 2)}')
    print('\nDistribuição de Classes:')
    print(df_class_dist.to_string(index=False))
    print('=' * 50)

    # Salvando em CSV
    if save_csv:
      metrics_csv_path = self.output_dir / 'corpus_metrics.csv'
      class_csv_path = self.output_dir / 'class_distribution.csv'

      metrics_summary.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
      df_class_dist.to_csv(class_csv_path, index=False, encoding='utf-8-sig')

      print(f'[INFO] Relatório de métricas salvo em: {metrics_csv_path}')
      print(f'[INFO] Relatório de classes salvo em: {class_csv_path}')

    return {
        'metrics_summary': metrics_summary,
        'class_distribution': df_class_dist,
    }

  def plot_class_distribution(self, save_path: str = None):
    """Gera e exibe o gráfico de barras da distribuição de classes."""
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=self.df,
        x=self.label_column,
        hue=self.label_column,
        palette='viridis',
        legend=False,
    )
    plt.title(
        'Distribuição de Classes do Corpus', fontsize=14, fontweight='bold'
    )
    plt.xlabel('Classe / Sentimento', fontsize=12)
    plt.ylabel('Quantidade de Avaliações', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path is None:
      save_path = self.output_dir / 'class_distribution.png'
    else:
      save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f'[INFO] Gráfico salvo em: {save_path}')
    plt.show()

  def generate_wordcloud(self, corpus_series: pd.Series, save_path: str = None):
    """Gera e exibe a nuvem de palavras com os termos de maior frequência."""
    all_text = ' '.join(corpus_series.astype(str))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis',
    ).generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(
        'Nuvem de Palavras do Corpus Pré-processado',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()

    if save_path is None:
      save_path = self.output_dir / 'wordcloud.png'
    else:
      save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f'[INFO] Nuvem de palavras salva em: {save_path}')
    plt.show()

  def suggest_unsupervised_applications(self) -> list:
    """Apresenta 3 opções práticas de aplicação não supervisionada."""
    applications = [
        {
            'opcao': (
                '1. Agrupamento de Avaliações (Clustering via K-Means ou'
                ' DBSCAN)'
            ),
            'descricao': (
                'Agrupar vetores de texto sem rótulos prévios para identificar'
                ' clusters temáticos recorrentes.'
            ),
        },
        {
            'opcao': (
                '2. Modelagem de Tópicos Latentes (LDA - Latent Dirichlet'
                ' Allocation)'
            ),
            'descricao': (
                'Extrair automaticamente tópicos latentes a partir da matriz'
                ' Bag-of-Words.'
            ),
        },
        {
            'opcao': '3. Redução de Dimensionalidade (t-SNE ou UMAP)',
            'descricao': (
                'Projetar os dados numéricos de alta dimensão em 2D/3D para'
                ' mapear visualmente a separabilidade natural.'
            ),
        },
    ]

    print('\n' + '=' * 50)
    print('APLICAÇÕES DE APRENDIZADO NÃO SUPERVISIONADO')
    print('=' * 50)
    for app in applications:
      print(f"\n• {app['opcao']}\n  {app['descricao']}")
    print('=' * 50)

    return applications