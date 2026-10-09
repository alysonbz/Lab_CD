import os
import urllib.request
from symspellpy import SymSpell, Verbosity

DICT_PATH = os.path.join("pre_process_and_analisis", "utils", "frequency_dictionary_pt.txt")

ABBREVIATIONS_MAP = {
    'pq': 'por que',
    'porq': 'porque',
    'oq': 'o que',
    'obg': 'obrigado',
    'obgd': 'obrigado',
    'obgda': 'obrigada',
    'tlg': 'estou ligado',
    'vc': 'voce',
    'vcs': 'voces',
    'tb': 'tambem',
    'tbm': 'tambem',
    'blz': 'beleza',
    'flw': 'falou',
    'vlw': 'valeu',
    'mto': 'muito',
    'mta': 'muita',
    'mtos': 'muitos',
    'mtas': 'muitas',
    'pfv': 'por favor',
    'pf': 'por favor',
    'kd': 'cade',
    'qdo': 'quando',
    'qto': 'quanto',
    'hj': 'hoje',
    'dps': 'depois',
    'tmj': 'estamos juntos',
    'add': 'adicionar',
    'att': 'atenciosamente',
    'msg': 'mensagem',
    'msgs': 'mensagens',
    'qtd': 'quantidade',
    'qtde': 'quantidade',
    'prod': 'produto',
    'prods': 'produtos',
}


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
                 "design", "bateria", "comprei", "recomendo", "entrega", "rapida", "processador", "wifi",
                 "gostei", "satisfeito", "insatisfeito", "otima", "otimo", "facil", "imagem", "imagens"]

    # Vocabulário crítico de e-commerce que recebe peso máximo
    critical_domain_words = {
        'gostei', 'satisfeito', 'insatisfeito', 'otima', 'otimo', 'otimas', 'otimos',
        'qualidade', 'preco', 'facil', 'bateria', 'imagem', 'imagens', 'display', 
        'design', 'wifi', 'wifis', 'gb', 'hd', 'led', 'produto', 'recomendo', 
        'desabone', 'rapida', 'bom', 'excelente', 'ruim'
    }

    print(f"[INFO] Escrevendo {len(words)} palavras no arquivo do SymSpell...")
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        for cw in critical_domain_words:
            f.write(f"{cw} 10000000\n")
            
        for w in words:
            word_clean = w.strip().lower()
            if word_clean and word_clean.isalpha() and word_clean not in critical_domain_words:
                f.write(f"{word_clean} 100000\n")

    print(f"[SUCESSO] Dicionário '{DICT_PATH}' pronto para uso!")
    return DICT_PATH


def get_pt_dictionary_path() -> str:
    """Apenas retorna o caminho do dicionário já garantido pelo processo principal."""
    if not os.path.exists(DICT_PATH) or os.path.getsize(DICT_PATH) == 0:
        ensure_pt_dictionary()
    return DICT_PATH


def worker_symspell_chunk(chunk: list) -> list:
    """
    Passe 1: Substituição de abreviações e correção ortográfica conservadora (Distância máxima 1).
    NUNCA altera palavras que já existem no dicionário nem palavras curtas/críticas.
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
    dict_path = get_pt_dictionary_path()
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    
    critical_domain_words = {
        'gostei', 'satisfeito', 'insatisfeito', 'otima', 'otimo', 'qualidade', 
        'preco', 'facil', 'bateria', 'imagem', 'imagens', 'display', 'design', 
        'wifi', 'wifis', 'gb', 'hd', 'led', 'produto', 'recomendo', 'desabone', 'rapida'
    }
    for word in critical_domain_words:
        sym_spell.create_dictionary_entry(word, 10000000)

    corrected_list = []
    for text in chunk:
        if not text:
            corrected_list.append("")
            continue
            
        words = text.split()
        corrected_words = []
        for word in words:
            word_lower = word.lower()

            # Regra 0: Expansão automática de abreviações (pq -> por que, obg -> obrigado, etc.)
            if word_lower in ABBREVIATIONS_MAP:
                corrected_words.append(ABBREVIATIONS_MAP[word_lower])
                continue

            # Regra 1: Palavras curtas (<=3) ou de domínio crítico não são alteradas
            if len(word) <= 3 or word_lower in critical_domain_words:
                corrected_words.append(word)
                continue
            
            # Regra 2: Se a palavra já existe no dicionário (distância 0), mantém intacta!
            exact_match = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=0)
            if exact_match:
                corrected_words.append(word)
            else:
                # Regra 3: Só corrige com distância máxima 1 se não for encontrada no dicionário
                suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=1)
                if suggestions:
                    corrected_words.append(suggestions[0].term)
                else:
                    corrected_words.append(word)
                    
        corrected_list.append(" ".join(corrected_words))
    return corrected_list


def worker_symspell_refine_chunk(args) -> list:
    """
    Passe 2: Refinamento conservador para palavras raras com suporte a expansão de abreviações.
    """
    if isinstance(args, tuple):
        chunk = args[0]
        rare_words = args[1] if len(args) > 1 else set()
    else:
        chunk = args
        rare_words = set()

    sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
    dict_path = get_pt_dictionary_path()
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    
    critical_domain_words = {
        'gostei', 'satisfeito', 'insatisfeito', 'otima', 'otimo', 'qualidade', 
        'preco', 'facil', 'bateria', 'imagem', 'imagens', 'display', 'design', 
        'wifi', 'wifis', 'gb', 'hd', 'led', 'produto', 'recomendo', 'desabone', 'rapida'
    }
    for word in critical_domain_words:
        sym_spell.create_dictionary_entry(word, 10000000)

    refined_list = []
    for text in chunk:
        if not text:
            refined_list.append("")
            continue
            
        words = text.split()
        corrected_words = []
        for word in words:
            word_lower = word.lower()

            # Regra 0: Expansão automática de abreviações
            if word_lower in ABBREVIATIONS_MAP:
                corrected_words.append(ABBREVIATIONS_MAP[word_lower])
                continue

            if word_lower in rare_words and len(word) > 3 and word_lower not in critical_domain_words:
                exact_match = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=0)
                if exact_match:
                    corrected_words.append(word)
                else:
                    suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=1)
                    if suggestions:
                        corrected_words.append(suggestions[0].term)
                    else:
                        corrected_words.append(word)
            else:
                corrected_words.append(word)
                
        refined_list.append(" ".join(corrected_words))
    return refined_list