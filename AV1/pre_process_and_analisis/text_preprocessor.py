import os
import pickle
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.sparse import issparse

from pre_process_and_analisis.text_cleaner import worker_clean_chunk
from pre_process_and_analisis.spell_checker import ensure_pt_dictionary, worker_symspell_chunk, worker_symspell_refine_chunk
from pre_process_and_analisis.lemmatizer import worker_lemmatize_chunk
from pre_process_and_analisis.eda_augmenter import worker_eda_chunk
from pre_process_and_analisis.smote_handler import handle_imbalance_smote


class TextPreprocessor:
    def __init__(
        self, 
        language: str = 'portuguese', 
        enable_spell_check: bool = True, 
        n_jobs: int = -1,
        cache_dir: str = 'pre_process_and_analisis/cache_checkpoint'
    ):
        self.language = language
        self.enable_spell_check = enable_spell_check
        self.cache_dir = cache_dir
        self._ensure_nltk_downloads()
        
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.enable_spell_check:
            ensure_pt_dictionary()
        
        if n_jobs == -1 or n_jobs is None:
            self.n_jobs = max(1, os.cpu_count() or 1)
        else:
            self.n_jobs = max(1, n_jobs)
            
        print(f"[INFO] TextPreprocessor ativo utilizando {self.n_jobs} núcleos do CPU.")
        print(f"[INFO] Pasta de checkpoints configurada: '{self.cache_dir}'")
        self.vectorizer_tfidf = None

    def _ensure_nltk_downloads(self):
        resources = ['punkt', 'stopwords', 'punkt_tab']
        for resource in resources:
            try:
                if resource in ['punkt', 'punkt_tab']:
                    nltk.data.find(f'tokenizers/{resource}')
                elif resource == 'stopwords':
                    nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def _parallel_execute(self, worker_fn, data_list: list, desc: str, task_id: str, extra_arg=None) -> list:
        total = len(data_list)
        if total == 0:
            return []
            
        n_chunks = min(total, self.n_jobs * 8)
        chunks = np.array_split(data_list, n_chunks)
        results = [None] * len(chunks)
        
        chunks_to_process = []
        cached_count = 0
        
        for idx, chunk in enumerate(chunks):
            chunk_file = os.path.join(self.cache_dir, f"{task_id}_chunk_{idx}.pkl")
            if os.path.exists(chunk_file):
                try:
                    with open(chunk_file, "rb") as f:
                        res = pickle.load(f)
                        if len(res) == len(chunk):
                            results[idx] = res
                            cached_count += len(chunk)
                        else:
                            chunks_to_process.append((idx, chunk))
                except Exception:
                    chunks_to_process.append((idx, chunk))
            else:
                chunks_to_process.append((idx, chunk))

        if cached_count > 0:
            print(f"[CACHE] Recuperado progresso salvo para '{task_id}': {cached_count}/{total} itens já processados.")

        if not chunks_to_process:
            print(f"[SUCESSO] Todos os chunks de '{task_id}' foram carregados do cache.")
            return [item for sublist in results for item in sublist]

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            for idx, chunk in chunks_to_process:
                chunk_list = chunk.tolist()
                arg = (chunk_list, extra_arg) if extra_arg is not None else chunk_list
                future = executor.submit(worker_fn, arg)
                futures[future] = (idx, chunk_list)

            with tqdm(total=total, desc=desc, unit="rev", initial=cached_count) as pbar:
                for future in as_completed(futures):
                    idx, chunk_list = futures[future]
                    res = future.result()
                    results[idx] = res
                    
                    chunk_file = os.path.join(self.cache_dir, f"{task_id}_chunk_{idx}.pkl")
                    with open(chunk_file, "wb") as f:
                        pickle.dump(res, f)
                        
                    try:
                        pkl_files = [
                            os.path.join(self.cache_dir, f) 
                            for f in os.listdir(self.cache_dir) 
                            if f.endswith('.pkl')
                        ]
                        if len(pkl_files) > 2:
                            pkl_files.sort(key=os.path.getmtime)
                            for old_file in pkl_files[:-2]:
                                os.remove(old_file)
                    except Exception:
                        pass
                        
                    pbar.update(len(chunk_list))

        return [item for sublist in results for item in sublist]

    def preprocess_corpus(self, series: pd.Series, task_prefix: str = "") -> pd.DataFrame:
        raw_texts = series.astype(str).tolist()
        
        # Step 1: Limpeza
        print(f"[INFO] Executando limpeza e normalização textual em paralelo{' (' + task_prefix.strip('_') + ')' if task_prefix else ''}...")
        cleaned = self._parallel_execute(
            worker_clean_chunk, 
            raw_texts, 
            desc=f"Limpeza de Textos ({task_prefix}p1)" if task_prefix else "Limpeza de Textos", 
            task_id=f"{task_prefix}step1_limpeza"
        )
        
        # Step 2: Correção Ortográfica (Dois Passes)
        if self.enable_spell_check:
            # Passe 1
            print("[INFO] Passe 1: Executando correção ortográfica SymSpell em paralelo...")
            cleaned = self._parallel_execute(
                worker_symspell_chunk, 
                cleaned, 
                desc=f"Correção Ortográfica (Passe 1 {task_prefix})".strip(), 
                task_id=f"{task_prefix}step2_spell_p1"
            )
            
            # Cálculo da frequência de palavras
            print("[INFO] Calculando frequência de palavras no corpus...")
            word_counts = Counter(word for text in cleaned for word in text.split())
            rare_words = {word for word, count in word_counts.items() if count == 1}
            print(f"[INFO] Identificadas {len(rare_words)} palavras raras (frequência = 1) para reanálise.")
            
            # Passe 2: Refinamento
            if len(rare_words) > 0:
                print("[INFO] Passe 2: Executando refinamento estendido para palavras de frequência = 1...")
                cleaned = self._parallel_execute(
                    worker_symspell_refine_chunk, 
                    cleaned, 
                    desc=f"Refinamento (Passe 2 {task_prefix})".strip(), 
                    task_id=f"{task_prefix}step2_spell_p2", 
                    extra_arg=rare_words
                )
            
        # Step 3: Tokenização e Lematização
        print("[INFO] Executando tokenização e lematização em paralelo...")
        processed = self._parallel_execute(
            worker_lemmatize_chunk, 
            cleaned, 
            desc=f"Tokenização & Lematização ({task_prefix}p3)" if task_prefix else "Tokenização & Lematização", 
            task_id=f"{task_prefix}step3_lemmatization", 
            extra_arg=self.language
        )

        return pd.DataFrame({
            'review_text_cleaned': cleaned,
            'review_text_processed': processed,
            'review_text_tokenized': [x.split() for x in processed]
        }, index=series.index)

    def transform_to_features(self, corpus: pd.Series, method: str = 'tfidf', max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer_tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        return self.vectorizer_tfidf.fit_transform(corpus)

    def generate_eda_dataset(self, df: pd.DataFrame, text_column: str, target_column: str, target_class: int, num_aug: int) -> pd.DataFrame:
        print("[INFO] Gerando aumento de dados textuais via EDA em paralelo...")
        df_eda = df.copy()
        df_eda['tipo_dado'] = 'original'
        
        minority_rows = df_eda[df_eda[target_column] == target_class]
        texts_to_aug = minority_rows[text_column].astype(str).tolist()
        
        augmented_results = self._parallel_execute(
            worker_eda_chunk, 
            texts_to_aug, 
            desc="Aumento EDA", 
            task_id="step_eda_aug", 
            extra_arg=num_aug
        )
        
        augmented_rows = []
        for (_, row), aug_texts in zip(minority_rows.iterrows(), augmented_results):
            for aug_text in aug_texts:
                new_row = row.copy()
                new_row[text_column] = aug_text
                new_row['tipo_dado'] = 'gerado_eda'
                augmented_rows.append(new_row)
                
        df_aug = pd.DataFrame(augmented_rows)
        df_final_eda = pd.concat([df_eda, df_aug], ignore_index=True)

        processed_cols = self.preprocess_corpus(df_final_eda[text_column], task_prefix="eda_")
        for col in processed_cols.columns:
            df_final_eda[col] = processed_cols[col]
            
        return df_final_eda

    def export_pipeline_results(
        self, 
        df_original: pd.DataFrame, 
        text_column: str = 'review_text', 
        target_column: str = 'polarity', 
        output_prefix: str = "dados",
        imbalance_threshold: float = 0.8
    ):
        output_dir = os.path.join("pre_process_and_analisis", "data")
        os.makedirs(output_dir, exist_ok=True)

        processed_filename = os.path.join(output_dir, f"{output_prefix}_preprocessados.csv")
        smote_filename = os.path.join(output_dir, f"{output_prefix}_aumentados_smote.csv")
        eda_filename = os.path.join(output_dir, f"{output_prefix}_aumentados_eda.csv")

        if os.path.exists(processed_filename) and os.path.exists(smote_filename) and os.path.exists(eda_filename):
            print(f"[CACHE] Datasets finais já encontrados em '{output_dir}'. Pulando pré-processamento e carregando dos arquivos CSV...")
            df_processed = pd.read_csv(processed_filename)
            df_smote = pd.read_csv(smote_filename)
            df_eda = pd.read_csv(eda_filename)
            return df_processed, df_smote, df_eda

        initial_len = len(df_original)
        df_original = df_original.dropna(subset=[target_column, text_column]).copy()
        dropped_len = initial_len - len(df_original)
        if dropped_len > 0:
            print(f"[AVISO] Foram removidas {dropped_len} linhas que continham valores nulos (NaN).")

        df_processed = df_original.copy()
        processed_df_cols = self.preprocess_corpus(df_processed[text_column])
        for col in processed_df_cols.columns:
            df_processed[col] = processed_df_cols[col]
            
        df_processed.to_csv(processed_filename, index=False)
        print(f"[SUCESSO] Dados pré-processados exportados para: {processed_filename}")
        
        y = df_processed[target_column].astype(int).values
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        print("\n" + "="*50)
        print(f"[INFO] ANÁLISE DE BALANCEAMENTO DA COLUNA '{target_column}'")
        print("="*50)
        for cls, count in zip(classes, counts):
            percentage = (count / total_samples) * 100
            print(f"   - Classe {cls}: {count} amostras ({percentage:.1f}%)")
        print("="*50)

        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = min_count / max_count if max_count > 0 else 1.0
        
        needs_balancing = (min_count < max_count) and (imbalance_ratio < imbalance_threshold)
        
        if not needs_balancing:
            print(f"[INFO] As classes já estão balanceadas (Razão min/máx = {imbalance_ratio:.2f}). Sem geração sintética.")

            df_smote = pd.DataFrame(
                self.transform_to_features(df_processed['review_text_processed'], method='tfidf', ngram_range=(1, 2)).toarray(),
                columns=[f"tfidf_{feat}" for feat in self.vectorizer_tfidf.get_feature_names_out()]
            )
            df_smote[target_column] = y
            df_smote['tipo_dado'] = 'original'
            
            df_eda = df_processed.copy()
            df_eda['tipo_dado'] = 'original'
            
            df_smote.to_csv(smote_filename, index=False)
            df_eda.to_csv(eda_filename, index=False)
            
            return df_processed, df_smote, df_eda

        minority_class = classes[np.argmin(counts)]
        n_needed = max_count - min_count
        
        print(f"[ALERTA] Desbalanceamento detectado! (Razão {imbalance_ratio:.2f} < limite {imbalance_threshold}).")
        print(f"   - Classe minoritária: {minority_class}")
        print(f"   - Serão geradas {n_needed} novas amostras para equilibrar las classes.\n")
        
        X_vec = self.transform_to_features(df_processed['review_text_processed'], method='tfidf', ngram_range=(1, 1), max_features=1500)
        X_res, y_res = handle_imbalance_smote(X_vec, y, target_class=minority_class, n_synthetic=n_needed)
        
        if issparse(X_res):
            X_res_dense = X_res.toarray()
        else:
            X_res_dense = X_res
            
        feature_names = [f"tfidf_{feat}" for feat in self.vectorizer_tfidf.get_feature_names_out()]
        df_smote = pd.DataFrame(X_res_dense, columns=feature_names)
        df_smote[target_column] = y_res
        
        num_originals = len(df_processed)
        df_smote['tipo_dado'] = ['original' if i < num_originals else 'gerado_smote' for i in range(len(df_smote))]
        
        df_smote.to_csv(smote_filename, index=False)
        print(f"[SUCESSO] Dataset SMOTE exportado para: {smote_filename}")

        num_aug_per_sample = int(np.ceil(n_needed / min_count))
        df_eda = self.generate_eda_dataset(
            df_original, 
            text_column=text_column, 
            target_column=target_column, 
            target_class=minority_class, 
            num_aug=num_aug_per_sample
        )
        
        df_eda.to_csv(eda_filename, index=False)
        print(f"[SUCESSO] Dataset EDA exportado para: {eda_filename}")

        return df_processed, df_smote, df_eda


if __name__ == "__main__":
    df = pd.read_csv("buscape.csv", encoding='utf-8', sep=',')

    preprocessor = TextPreprocessor(
        language='portuguese', 
        enable_spell_check=True, 
        n_jobs=-1,
        cache_dir='pre_process_and_analisis/cache_checkpoint'
    )

    df_proc, df_smote, df_eda = preprocessor.export_pipeline_results(
        df_original=df, 
        text_column='review_text', 
        target_column='polarity',
        output_prefix='reviews_ecommerce'
    )