import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class CorpusExploratoryAnalyzer:
    """Classe focada em análise exploratória de dados (EDA), métricas estatísticas,
    visualizações gráficas comparativas entre datasets, agrupamento não supervisionado (K-Means e LDA)
    e exportação de relatórios.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        text_column: str = 'review_text_cleaned',
        label_column: str = 'polarity',
        output_dir: str = None,
    ):
        self.df = df
        self.text_column = text_column
        self.label_column = label_column

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).resolve().parent / 'visualizations'

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_datasets(data_dir: str = "pre_process_and_analisis/data", prefix: str = "reviews_ecommerce") -> dict:
        """Carrega os datasets pré-processados e aumentados (Original, SMOTE, EDA) da pasta informada."""
        path_orig = os.path.join(data_dir, f"{prefix}_preprocessados.csv")
        path_smote = os.path.join(data_dir, f"{prefix}_aumentados_smote.csv")
        path_eda = os.path.join(data_dir, f"{prefix}_aumentados_eda.csv")

        datasets = {}
        for name, path in [("Original", path_orig), ("SMOTE", path_smote), ("EDA", path_eda)]:
            if os.path.exists(path):
                print(f"[INFO] Carregando dataset {name}: '{path}'...")
                datasets[name] = pd.read_csv(path)
            else:
                print(f"[AVISO] Dataset {name} não encontrado no caminho '{path}'.")
        return datasets

    def get_basic_metrics(self, df: pd.DataFrame = None, text_column: str = None, label_column: str = None, save_csv: bool = True) -> dict:
        """Calcula métricas descritivas do corpus e exporta os relatórios para CSV."""
        target_df = df if df is not None else self.df
        t_col = text_column or self.text_column
        l_col = label_column or self.label_column

        if target_df is None or target_df.empty:
            print("[ERRO] Nenhum DataFrame fornecido para calcular métricas.")
            return {}

        total_docs = len(target_df)
        num_classes = target_df[l_col].nunique()
        class_distribution = target_df[l_col].value_counts()

        if t_col in target_df.columns:
            text_series = target_df[t_col].fillna('').astype(str)
            char_lengths = text_series.apply(len)
            word_counts = text_series.apply(lambda x: len(x.split()))
            null_count = target_df[t_col].isnull().sum()
            
            mean_chars = round(float(char_lengths.mean()), 2)
            min_chars, max_chars = int(char_lengths.min()), int(char_lengths.max())
            mean_words = round(float(word_counts.mean()), 2)
            min_words, max_words = int(word_counts.min()), int(word_counts.max())
            total_words = int(word_counts.sum())
        else:
            null_count = 0
            mean_chars = min_chars = max_chars = 0
            mean_words = min_words = max_words = total_words = 0

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
                total_docs, num_classes, null_count,
                mean_chars, min_chars, max_chars,
                mean_words, min_words, max_words, total_words
            ],
        })

        df_class_dist = pd.DataFrame({
            'classe': class_distribution.index,
            'quantidade': class_distribution.values,
            'percentual': np.round((class_distribution.values / total_docs) * 100, 2),
        })

        print('=' * 50)
        print('MÉTRICAS BÁSICAS E DESCRITIVAS DO CORPUS')
        print('=' * 50)
        print(f'• Tamanho total do corpus: {total_docs} documentos')
        print(f'• Número de classes únicas: {num_classes}')
        print(f'• Média de palavras/doc: {mean_words}')
        print(f'• Média de caracteres/doc: {mean_chars}')
        print('\nDistribuição de Classes:')
        print(df_class_dist.to_string(index=False))
        print('=' * 50)

        if save_csv:
            metrics_csv_path = self.output_dir / 'corpus_metrics.csv'
            class_csv_path = self.output_dir / 'class_distribution.csv'
            metrics_summary.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
            df_class_dist.to_csv(class_csv_path, index=False, encoding='utf-8-sig')

        return {'metrics_summary': metrics_summary, 'class_distribution': df_class_dist}

    def plot_class_distribution(self, save_path: str = None):
        """Gera e salva o gráfico de barras da distribuição de classes do corpus."""
        if self.df is None or self.df.empty:
            return

        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=self.df,
            x=self.label_column,
            hue=self.label_column,
            palette='viridis',
            legend=False,
        )
        plt.title('Distribuição de Classes do Corpus', fontsize=14, fontweight='bold')
        plt.xlabel('Classe / Sentimento', fontsize=12)
        plt.ylabel('Quantidade de Avaliações', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = Path(save_path) if save_path else self.output_dir / 'class_distribution.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f'[INFO] Gráfico de distribuição de classes salvo em: {save_path}')
        plt.close()

    def plot_datasets_class_comparison(self, datasets: dict, label_column: str = 'polarity', save_path: str = None):
        """Gera um gráfico comparativo de barras com o número de amostras por classe em cada dataset."""
        data = []
        for name, df_item in datasets.items():
            if label_column in df_item.columns:
                counts = df_item[label_column].value_counts()
                for cls, count in counts.items():
                    data.append({'Dataset': name, 'Classe': str(cls), 'Quantidade': count})

        df_comp = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_comp, x='Dataset', y='Quantidade', hue='Classe', palette='viridis')
        
        plt.title('Número de Amostras por Classe (Original vs SMOTE vs EDA)', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Quantidade de Avaliações', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f'{int(height)}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                            textcoords='offset points')

        plt.tight_layout()
        save_path = Path(save_path) if save_path else self.output_dir / 'comparison_class_distribution.png'
        plt.savefig(save_path, dpi=300)
        print(f'[INFO] Gráfico comparativo de classes salvo em: {save_path}')
        plt.close()

    def plot_top_words_per_class(self, datasets: dict, text_column: str = 'review_text_cleaned', label_column: str = 'polarity', top_n: int = 10):
        """Gera gráficos ordenados das palavras/features mais frequentes para cada classe em cada dataset."""
        for ds_name, df_item in datasets.items():
            if label_column not in df_item.columns:
                continue

            classes = sorted(df_item[label_column].unique())
            fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 5), sharey=False)
            if len(classes) == 1:
                axes = [axes]

            fig.suptitle(f'Top {top_n} Palavras Mais Frequentes por Classe - Dataset {ds_name}', fontsize=14, fontweight='bold')

            for idx, cls in enumerate(classes):
                df_cls = df_item[df_item[label_column] == cls]

                if text_column in df_cls.columns:
                    all_words = " ".join(df_cls[text_column].fillna("").astype(str)).split()
                    counts = Counter(all_words).most_common(top_n)
                    words, freqs = zip(*counts) if counts else ([], [])
                else:
                    tfidf_cols = [c for c in df_cls.columns if c.startswith('tfidf_')]
                    if tfidf_cols:
                        sums = df_cls[tfidf_cols].sum(axis=0).sort_values(ascending=False).head(top_n)
                        words = [c.replace('tfidf_', '') for c in sums.index]
                        freqs = sums.values
                    else:
                        words, freqs = [], []

                if words:
                    df_words = pd.DataFrame({'Palavra': words, 'Frequência': freqs})
                    sns.barplot(data=df_words, x='Frequência', y='Palavra', ax=axes[idx], palette='crest')
                    axes[idx].set_title(f'Classe: {cls}', fontsize=12, fontweight='bold')
                    axes[idx].set_xlabel('Frequência / Peso Sumarizado')
                    axes[idx].set_ylabel('')
                    axes[idx].grid(axis='x', linestyle='--', alpha=0.7)

            plt.tight_layout()
            save_path = self.output_dir / f'top_words_{ds_name.lower()}.png'
            plt.savefig(save_path, dpi=300)
            print(f'[INFO] Gráfico de palavras mais frequentes ({ds_name}) salvo em: {save_path}')
            plt.close(fig)

    def generate_wordcloud(self, corpus_series: pd.Series, save_path: str = None):
        """Gera e salva a nuvem de palavras com os termos de maior frequência de forma limpa."""
        texts = corpus_series.dropna().astype(str).tolist()
        cleaned_texts = [" ".join(set(t.split())) for t in texts]
        all_text = ' '.join(cleaned_texts)

        wordcloud = WordCloud(
            width=800, height=400, background_color='white', max_words=100, colormap='viridis'
        ).generate(all_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nuvem de Palavras do Corpus Pré-processado', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = Path(save_path) if save_path else self.output_dir / 'wordcloud.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f'[INFO] Nuvem de palavras salva em: {save_path}')
        plt.close()

    def apply_unsupervised_analysis(
        self,
        df: pd.DataFrame = None,
        text_column: str = None,
        n_clusters: int = 4,
        top_n_words: int = 6,
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Aplica K-Means e LDA no corpus, rotula os documentos com base nas palavras mais
        representativas de cada grupo e gera gráficos explicativos e relatórios CSV.
        """
        target_df = (df if df is not None else self.df).copy()
        t_col = text_column or self.text_column

        if target_df is None or t_col not in target_df.columns:
            print(f"[ERRO] Coluna '{t_col}' não encontrada para agrupamento não supervisionado.")
            return target_df

        text_series = target_df[t_col].fillna("").astype(str)

        print("\n" + "="*60)
        print(f"[INFO] EXECUTANDO AGRUPAMENTO NÃO SUPERVISIONADO (K-Means & LDA - {n_clusters} Grupos)")
        print("="*60)

        # 1. Vetorização TF-IDF para K-Means e CountVectorizer para LDA
        tfidf_vec = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf_vec.fit_transform(text_series)
        tfidf_features = np.array(tfidf_vec.get_feature_names_out())

        count_vec = CountVectorizer(max_features=1000)
        X_count = count_vec.fit_transform(text_series)
        count_features = np.array(count_vec.get_feature_names_out())

        # 2. K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        target_df['kmeans_cluster'] = kmeans.fit_predict(X_tfidf)

        kmeans_labels = {}
        for i, center in enumerate(kmeans.cluster_centers_):
            top_indices = center.argsort()[-top_n_words:][::-1]
            top_words = ", ".join(tfidf_features[top_indices])
            kmeans_labels[i] = f"Cluster {i}: [{top_words}]"

        target_df['kmeans_label'] = target_df['kmeans_cluster'].map(kmeans_labels)

        # 3. LDA (Latent Dirichlet Allocation)
        lda = LatentDirichletAllocation(n_components=n_clusters, random_state=42)
        lda_probs = lda.fit_transform(X_count)
        target_df['lda_topic'] = lda_probs.argmax(axis=1)

        lda_labels = {}
        for i, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-top_n_words:][::-1]
            top_words = ", ".join(count_features[top_indices])
            lda_labels[i] = f"Tópico {i}: [{top_words}]"

        target_df['lda_label'] = target_df['lda_topic'].map(lda_labels)

        # 4. Visualização 1: Distribuição dos Grupos K-Means e Tópicos LDA
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.countplot(data=target_df, y='kmeans_label', ax=axes[0], palette='viridis')
        axes[0].set_title('K-Means: Distribuição de Avaliações por Grupo', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Quantidade de Avaliações')
        axes[0].set_ylabel('')

        sns.countplot(data=target_df, y='lda_label', ax=axes[1], palette='magma')
        axes[1].set_title('LDA: Distribuição de Avaliações por Tópico', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Quantidade de Avaliações')
        axes[1].set_ylabel('')

        plt.tight_layout()
        plot_path = self.output_dir / 'unsupervised_clusters_distribution.png'
        plt.savefig(plot_path, dpi=300)
        print(f"[INFO] Gráfico de distribuição de clusters salvo em: {plot_path}")
        plt.close(fig)

        # 5. Visualização 2: Projeção 2D dos Clusters K-Means via PCA
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(X_tfidf.toarray())
        pca_df = pd.DataFrame({
            'PCA1': coords_2d[:, 0],
            'PCA2': coords_2d[:, 1],
            'Grupo': target_df['kmeans_label']
        })

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Grupo', palette='tab10', alpha=0.5, s=25)
        plt.title('Projeção 2D dos Clusters K-Means (PCA)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pca_path = self.output_dir / 'kmeans_pca_scatter.png'
        plt.savefig(pca_path, dpi=300)
        print(f"[INFO] Projeção PCA de clusters salva em: {pca_path}")
        plt.close()

        # 6. Exportação de Relatórios em CSV
        if save_csv:
            csv_path = self.output_dir / 'unsupervised_clustering_results.csv'
            target_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"[SUCESSO] Dataset rotulado não supervisionado exportado para: {csv_path}")

            summary_rows = []
            for i in range(n_clusters):
                summary_rows.append({
                    'ID_Grupo': i,
                    'KMeans_TopPalavras': kmeans_labels[i],
                    'KMeans_Qtd_Docs': (target_df['kmeans_cluster'] == i).sum(),
                    'LDA_TopPalavras': lda_labels[i],
                    'LDA_Qtd_Docs': (target_df['lda_topic'] == i).sum(),
                })
            summary_df = pd.DataFrame(summary_rows)
            summary_csv_path = self.output_dir / 'unsupervised_summary.csv'
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"[SUCESSO] Resumo dos agrupamentos exportado para: {summary_csv_path}")

        return target_df


if __name__ == '__main__':
    datasets = CorpusExploratoryAnalyzer.load_datasets()

    if datasets:
        analyzer = CorpusExploratoryAnalyzer(output_dir="pre_process_and_analisis/visualizations")

        if "Original" in datasets:
            analyzer.get_basic_metrics(df=datasets["Original"], text_column="review_text_cleaned", label_column="polarity")
            analyzer.generate_wordcloud(datasets["Original"]["review_text_cleaned"])

            analyzer.apply_unsupervised_analysis(
                df=datasets["Original"],
                text_column="review_text_cleaned",
                n_clusters=4
            )

        analyzer.plot_datasets_class_comparison(datasets, label_column="polarity")
        analyzer.plot_top_words_per_class(datasets, text_column="review_text_cleaned", label_column="polarity", top_n=10)