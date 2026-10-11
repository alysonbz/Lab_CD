import os
import pickle
import time
from random import uniform
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

class LocalBackTranslationAugmenter:
    """Realiza aumento de dados por tradução reversa utilizando lote (translate_batch) 
    para alta velocidade, com proteções anti-rate-limit e salvamento incremental.
    """
    def __init__(self, cache_dir: str = 'pre_process_and_analisis/cache_checkpoint'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print("[INFO] Tradutor por API otimizado com lote (translate_batch) e checkpointing.")

    def back_translate_batch(self, texts: list, batch_size: int = 50, retries: int = 3) -> list:
        """Executa a ida (PT -> EN) e a volta (EN -> PT) em lotes para máxima performance."""
        back_translated_texts = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            success = False
            
            for attempt in range(retries):
                try:
                    # 1. Ida: Português para Inglês em lote
                    en_batch = GoogleTranslator(source='pt', target='en').translate_batch(batch)
                    if not en_batch:
                        en_batch = batch
                        
                    # Pequena pausa de respiro entre a ida e a volta
                    time.sleep(uniform(0.1, 0.3))
                    
                    # 2. Volta: Inglês para Português em lote
                    pt_batch = GoogleTranslator(source='en', target='portuguese').translate_batch(en_batch)
                    if not pt_batch:
                        pt_batch = en_batch
                        
                    # Consolida os resultados garantindo fallback caso algum item venha nulo
                    final_batch = [p if p else o for p, o in zip(pt_batch, batch)]
                    back_translated_texts.extend(final_batch)
                    success = True
                    break
                    
                except Exception as e:
                    if attempt < retries - 1:
                        sleep_time = (2 ** attempt) * 2  # Backoff exponencial (2s, 4s, 8s)
                        time.sleep(sleep_time)
                    else:
                        # Se esgotar as tentativas, mantém o texto original para não perder dados
                        back_translated_texts.extend(batch)
                        success = True
                        
            # Pausa rápida e segura entre os blocos
            time.sleep(uniform(0.2, 0.4))
            
        return back_translated_texts

    def augment_minority_class(self, df: pd.DataFrame, text_column: str, target_column: str, target_class: int, n_needed: int) -> pd.DataFrame:
        df_aug = df.copy()
        if 'tipo_dado' not in df_aug.columns:
            df_aug['tipo_dado'] = 'original'
            
        minority_rows = df_aug[df_aug[target_column] == target_class]
        if len(minority_rows) == 0:
            return df_aug
            
        sampled_rows = minority_rows.sample(n=n_needed, replace=True, random_state=42)
        texts_to_translate = sampled_rows[text_column].astype(str).tolist()
        
        cache_file = os.path.join(self.cache_dir, "back_translation_progress.pkl")
        generated_texts = []
        start_idx = 0
        
        # Recupera o cache anterior caso o script tenha sido interrompido
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    generated_texts = pickle.load(f)
                    start_idx = len(generated_texts)
                    print(f"[CACHE] Progresso anterior encontrado! Retomando a partir da amostra {start_idx}/{len(texts_to_translate)}.")
            except Exception:
                generated_texts = []
                start_idx = 0

        remaining_texts = texts_to_translate[start_idx:]
        
        if len(remaining_texts) > 0:
            print(f"[INFO] Iniciando tradução em lote via API para {len(remaining_texts)} amostras restantes (Total: {len(texts_to_translate)})...")
            
            # Processa em blocos de 100 amostras por ciclo de salvamento no checkpoint
            chunk_size = 100
            with tqdm(total=len(remaining_texts), desc="Back-Translation Batch API", unit="amostra") as pbar:
                for j in range(0, len(remaining_texts), chunk_size):
                    chunk = remaining_texts[j:j + chunk_size]
                    
                    # Traduz o bloco utilizando lotes internos de 50
                    translated_chunk = self.back_translate_batch(chunk, batch_size=50)
                    generated_texts.extend(translated_chunk)
                    
                    # Salva o checkpoint imediatamente após cada bloco
                    with open(cache_file, "wb") as f:
                        pickle.dump(generated_texts, f)
                        
                    pbar.update(len(chunk))
        
        augmented_rows = []
        for (_, row), new_text in zip(sampled_rows.iterrows(), generated_texts):
            if new_text and new_text.strip():
                new_row = row.copy()
                new_row[text_column] = new_text
                new_row['tipo_dado'] = 'gerado_traducao'
                augmented_rows.append(new_row)
                
        df_augmented_final = pd.concat([df_aug, pd.DataFrame(augmented_rows)], ignore_index=True)
        print(f"[SUCESSO] Aumento por tradução concluído. Total: {len(df_augmented_final)}")
        
        # Limpa o arquivo de checkpoint após concluir com sucesso
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass
                
        return df_augmented_final