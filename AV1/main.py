from pathlib import Path
import sys

from pre_process_and_analisis.analise_and_pre_process import run_pipeline

def main():
    """Arquivo central da aplicação responsável estritamente por orquestrar os módulos."""
    DATASET_PATH = 'buscape.csv'
    
    print("=== INICIANDO PIPELINE DO PROJETO DE NLP (BUSCAPÉ) ===")
   
    X_processed, y_processed, raw_df = run_pipeline(DATASET_PATH)
    
    print("=== ETAPAS 1 E 2 EXECUTADAS COM SUCESSO ===")

if __name__ == "__main__":
    main()