import os
import torch
from transformers import MarianTokenizer, MarianMTModel
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

class LocalBackTranslationAugmenter:
    """Realiza aumento de dados por tradução reversa utilizando o modelo unificado MarianMT 
    (Helsinki-NLP/opus-mt-tc-big-en-pt) para ida e volta, com barra de progresso e token do .env.
    """
    def __init__(self, device: str = None):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Pega o token do .env
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY")
        if not self.hf_token:
            print("[AVISO] HUGGINGFACE_API_KEY não encontrada no arquivo .env. Tentando acesso público...")

        print(f"[INFO] Carregando modelo unificado de tradução local no dispositivo: {self.device}")
        
        # Modelo oficial unificado do Tatoeba Challenge para o par EN-PT
        self.model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        
        print("[INFO] Carregando tokenizer e modelo MarianMT...")
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        self.model = MarianMTModel.from_pretrained(self.model_name, token=self.hf_token).to(self.device)

    def translate(self, texts: list, target_lang: str, batch_size: int = 16, desc: str = "Traduzindo") -> list:
        """Traduz os textos injetando o prefixo do idioma de destino com barra de progresso (tqdm)."""
        translated_texts = []
        prefixed_batch = [f"{target_lang} {t}" for t in texts]
        
        total_batches = (len(prefixed_batch) + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc=desc, unit="lote") as pbar:
            for i in range(0, len(prefixed_batch), batch_size):
                batch = prefixed_batch[i:i + batch_size]
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                
                with torch.no_grad():
                    translated_tokens = self.model.generate(**encoded, max_length=512, num_beams=4)
                    
                decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translated_texts.extend(decoded)
                pbar.update(1)
                
        return translated_texts

    def back_translate(self, texts: list, batch_size: int = 16) -> list:
        """Executa a ida (PT -> EN) e a volta (EN -> PT) com barras de progresso dedicadas."""
        if not texts:
            return []
            
        print("[INFO] Traduzindo de Português para Inglês (Ida)...")
        english_batch = self.translate(texts, target_lang=">>en<<", batch_size=batch_size, desc="PT -> EN")
        
        print("[INFO] Traduzindo de volta para Português (Volta/Variação)...")
        back_translated_batch = self.translate(english_batch, target_lang=">>por<<", batch_size=batch_size, desc="EN -> PT")
        
        return back_translated_batch

    def augment_minority_class(self, df: pd.DataFrame, text_column: str, target_column: str, target_class: int, n_needed: int) -> pd.DataFrame:
        df_aug = df.copy()
        if 'tipo_dado' not in df_aug.columns:
            df_aug['tipo_dado'] = 'original'
            
        minority_rows = df_aug[df_aug[target_column] == target_class]
        if len(minority_rows) == 0:
            return df_aug
            
        sampled_rows = minority_rows.sample(n=min(n_needed, len(minority_rows)), replace=True, random_state=42)
        texts_to_translate = sampled_rows[text_column].astype(str).tolist()
        
        print(f"[INFO] Iniciando tradução reversa para gerar {len(texts_to_translate)} novas amostras sintéticas...")
        generated_texts = self.back_translate(texts_to_translate, batch_size=16)
        
        augmented_rows = []
        for (_, row), new_text in zip(sampled_rows.iterrows(), generated_texts):
            if new_text and new_text.strip():
                new_row = row.copy()
                new_row[text_column] = new_text
                new_row['tipo_dado'] = 'gerado_traducao'
                augmented_rows.append(new_row)
                
        df_augmented_final = pd.concat([df_aug, pd.DataFrame(augmented_rows)], ignore_index=True)
        print(f"[SUCESSO] Aumento por tradução concluído. Total de registros resultante: {len(df_augmented_final)}")
        
        return df_augmented_final