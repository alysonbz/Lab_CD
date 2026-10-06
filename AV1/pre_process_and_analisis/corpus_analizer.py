import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class CorpusExploratoryAnalyzer:
    """
    Classe focada em análise exploratória de dados (EDA), métricas estatísticas,
    visualizações gráficas e proposição de tarefas não supervisionadas.
    """
    def __init__(self, df: pd.DataFrame, text_column: str, label_column: str):
        self.df = df
        self.text_column = text_column
        self.label_column = label_column

    def get_basic_metrics(self) -> dict:
        """Calcula e exibe métricas descritivas do corpus."""
        total_docs = len(self.df)
        num_classes = self.df[self.label_column].nunique()
        class_distribution = self.df[self.label_column].value_counts()
        
        print("=" * 50)
        print("MÉTRICAS BÁSICAS DO CORPUS")
        print("=" * 50)
        print(f"• Tamanho total do corpus: {total_docs} documentos")
        print(f"• Número de classes únicas: {num_classes}")
        print("\nDistribuição de Classes:")
        print(class_distribution)
        print("=" * 50)

        return {
            "total_docs": total_docs,
            "num_classes": num_classes,
            "class_distribution": class_distribution
        }

    def plot_class_distribution(self, save_path: str = None):
        """Gera e exibe o gráfico de barras da distribuição de classes."""
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x=self.label_column, palette='viridis')
        plt.title('Distribuição de Classes do Corpus', fontsize=14, fontweight='bold')
        plt.xlabel('Classe / Sentimento', fontsize=12)
        plt.ylabel('Quantidade de Avaliações', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Gráfico salvo em: {save_path}")
        plt.show()

    def generate_wordcloud(self, corpus_series: pd.Series, save_path: str = None):
        """Gera e exibe a nuvem de palavras com os termos de maior frequência."""
        all_text = " ".join(corpus_series.astype(str))
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            max_words=100, 
            colormap='viridis'
        ).generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nuvem de Palavras do Corpus Pré-processado', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Nuvem de palavras salva em: {save_path}")
        plt.show()

    def suggest_unsupervised_applications(self) -> list:
        """Apresenta 3 opções práticas de aplicação não supervisionada."""
        applications = [
            {
                "opcao": "1. Agrupamento de Avaliações (Clustering via K-Means ou DBSCAN)",
                "descricao": "Agrupar vetores de texto sem rótulos prévios para identificar clusters temáticos recorrentes."
            },
            {
                "opcao": "2. Modelagem de Tópicos Latentes (LDA - Latent Dirichlet Allocation)",
                "descricao": "Extrair automaticamente tópicos latentes a partir da matriz Bag-of-Words."
            },
            {
                "opcao": "3. Redução de Dimensionalidade (t-SNE ou UMAP)",
                "descricao": "Projetar os dados numéricos de alta dimensão em 2D/3D para mapear visualmente a separabilidade natural."
            }
        ]
        
        print("\n" + "=" * 50)
        print("APLICAÇÕES DE APRENDIZADO NÃO SUPERVISIONADO")
        print("=" * 50)
        for app in applications:
            print(f"\n• {app['opcao']}\n  {app['descricao']}")
        print("=" * 50)

        return applications