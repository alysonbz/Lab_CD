import os
import urllib.request
from symspellpy import SymSpell, Verbosity

DICT_PATH = os.path.join("pre_process_and_analisis", "utils", "frequency_dictionary_pt.txt")

def ensure_pt_dictionary() -> str:
    """
    Garante a criação do dicionário no PROCESSO PRINCIPAL (thread única).
    Evita race conditions em ambientes multi-processo.
    """
    os.makedirs(os.path.dirname(DICT_PATH), exist_ok=True)
    
    if os.path.exists(DICT_PATH) and os.path.getsize(DICT_PATH) > 0:
        return DICT_PATH

    print(f"[INFO] Criando dicionário '{DICT_PATH}' a partir do repositório pythonprobr/palavras...")
    urls = [
        "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt",
        "https://raw.githubusercontent.com/pythonprobr/palavras/main/palavras.txt"
    ]
    
    words = []
    download_success = False
    
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8', errors='ignore')
                words = content.splitlines()
                if words:
                    download_success = True
                    break
        except Exception:
            continue

    if not download_success or not words:
        print("[AVISO] Não foi possível conectar ao GitHub. Gerando dicionário base offline...")
        words = ["o", "a", "que", "e", "do", "da", "em", "um", "para", "com", "nao", "uma", "os", "no", 
                 "se", "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele", "das", "seu", "sua",
                 "produto", "excelente", "muito", "bom", "ruim", "qualidade", "preco", "teclado", "display", 
                 "design", "bateria", "comprei", "recomendo", "entrega", "rapida", "processador", "wifi"]

    print(f"[INFO] Escrevendo {len(words)} palavras no arquivo do SymSpell...")
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        for w in words:
            word_clean = w.strip().lower()
            if word_clean and word_clean.isalpha():
                f.write(f"{word_clean} 10000\n")

    print(f"[SUCESSO] Dicionário '{DICT_PATH}' pronto para uso!")
    return DICT_PATH


def get_pt_dictionary_path() -> str:
    """Apenas retorna o caminho do dicionário já garantido pelo processo principal."""
    if not os.path.exists(DICT_PATH) or os.path.getsize(DICT_PATH) == 0:
        ensure_pt_dictionary()
    return DICT_PATH


def worker_symspell_chunk(chunk: list) -> list:
    """Passe 1: Correção ortográfica inicial ultra-rápida (Distância 2)."""
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = get_pt_dictionary_path()
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    
    custom_words = ['display', 'design', 'wifi', 'wifis', 'gb', 'hd', 'led']
    for word in custom_words:
        sym_spell.create_dictionary_entry(word, 1000000)

    corrected_list = []
    for text in chunk:
        if not text:
            corrected_list.append("")
            continue
            
        words = text.split()
        corrected_words = []
        for word in words:
            if len(word) > 3:
                suggestions = sym_spell.lookup(
                    word, 
                    Verbosity.TOP, 
                    max_edit_distance=2
                )
                if suggestions:
                    corrected_words.append(suggestions[0].term)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        corrected_list.append(" ".join(corrected_words))
    return corrected_list


def worker_symspell_refine_chunk(args: tuple) -> list:
    """Passe 2: Refinamento direcionado para palavras com frequência == 1 no corpus."""
    chunk, rare_words = args
    
    sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
    dict_path = get_pt_dictionary_path()
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    
    custom_words = ['display', 'design', 'wifi', 'wifis', 'gb', 'hd', 'led']
    for word in custom_words:
        sym_spell.create_dictionary_entry(word, 1000000)

    refined_list = []
    for text in chunk:
        if not text:
            refined_list.append("")
            continue
            
        words = text.split()
        corrected_words = []
        for word in words:
            if word in rare_words and len(word) > 3:
                suggestions = sym_spell.lookup(
                    word, 
                    Verbosity.ALL, 
                    max_edit_distance=3
                )
                if suggestions:
                    best_candidate = suggestions[0]
                    
                    if best_candidate.distance == 0 and len(suggestions) > 1:
                        for cand in suggestions[1:]:
                            if cand.distance <= 2 and cand.count > (best_candidate.count * 100):
                                best_candidate = cand
                                break
                                
                    elif best_candidate.distance > 0:
                        valid_cands = [s for s in suggestions if s.distance <= 3]
                        if valid_cands:
                            best_candidate = max(valid_cands, key=lambda s: (s.count / (s.distance * 10)))
                            
                    corrected_words.append(best_candidate.term)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
                
        refined_list.append(" ".join(corrected_words))
    return refined_list