import os
import torch
from transformers import MarianTokenizer, MarianMTModel
import pandas as pd
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

class LocalBackTranslationAugmenter:
    """Realiza aumento de dados por tradução reversa utilizando modelos MarianMT locais,
    lendo o token de autenticação de forma segura a partir de um arquivo .env.
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

        print(f"[INFO] Carregando modelos de tradução local no dispositivo: {self.device}")
        
        # Identificadores oficiais dos modelos Marian para o par Português <-> Inglês
        self.pt_to_en_model_name = "Helsinki-NLP/opus-mt-pt-en"
        self.en_to_pt_model_name = "Helsinki-NLP/opus-mt-en-pt"
        
        print("[INFO] Carregando tradutor PT -> EN...")
        self.tokenizer_pt_en = MarianTokenizer.from_pretrained(self.pt_to_en_model_name, token=self.hf_token)
        self.model_pt_en = MarianMTModel.from_pretrained(self.pt_to_en_model_name, token=self.hf_token).to(self.device)
        
        print("[INFO] Carregando tradutor EN -> PT...")
        self.tokenizer_en_pt = MarianTokenizer.from_pretrained(self.en_to_pt_model_name, token=self.hf_token)
        self.model_en_pt = MarianMTModel.from_pretrained(self.en_to_pt_model_name, token=self.hf_token).to(self.device)

    def translate(self, texts: list, model, tokenizer, batch_size: int = 16) -> list:
        translated_texts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                translated_tokens = model.generate(**encoded, max_length=512, num_beams=4)
                
            decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated_texts.extend(decoded)
            
        return translated_texts

    def back_translate(self, texts: list, batch_size: int = 16) -> list:
        """Executa a ida (PT -> EN) e a volta (EN -> PT)."""
        if not texts:
            return []
            
        print("[INFO] Traduzindo de Português para Inglês...")
        english_batch = self.translate(texts, self.model_pt_en, self.tokenizer_pt_en, batch_size)
        
        print("[INFO] Traduzindo de volta para Português (Gerando variações)...")
        back_translated_batch = self.translate(english_batch, self.model_en_pt, self.tokenizer_en_pt, batch_size)
        
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