import numpy as np

def worker_eda_chunk(args: tuple) -> list:
    """Worker para aumento de dados textuais via EDA (Easy Data Augmentation) robusto e diversificado."""
    chunk_texts, num_aug = args
    augmented_results = []
    
    for text in chunk_texts:
        words = text.split()
        if not words or len(words) <= 3:
            # Se a frase for muito curta, duplica com pequenas variações de espaçamento ou retorna o original com fallback
            augmented_results.append([text] * num_aug)
            continue
            
        aug_list = []
        seen_variants = {text} # Evita duplicatas exatas para a mesma frase
        
        attempts = 0
        while len(aug_list) < num_aug and attempts < num_aug * 3:
            attempts += 1
            new_words = words.copy()
            
            # Escolhe aleatoriamente uma técnica do EDA (0: Swap, 1: Random Deletion, 2: Random Insertion)
            op_type = np.random.choice([0, 1, 2])
            
            if op_type == 0 and len(new_words) >= 2:
                # 1. Random Swap (Troca de posições)
                idx1, idx2 = np.random.choice(len(new_words), 2, replace=False)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
                
            elif op_type == 1 and len(new_words) > 4:
                # 2. Random Deletion (Remove 1 a 2 palavras aleatórias para dar fluidez)
                n_to_delete = np.random.randint(1, min(3, len(new_words) - 2))
                indices_to_del = np.random.choice(len(new_words), n_to_delete, replace=False)
                new_words = [w for i, w in enumerate(new_words) if i not in indices_to_del]
                
            elif op_type == 2 and len(new_words) >= 3:
                # 3. Random Insertion (Insere uma palavra existente do próprio texto em outra posição)
                random_word = np.random.choice(new_words)
                random_pos = np.random.randint(0, len(new_words))
                new_words.insert(random_pos, random_word)
                
            new_text = " ".join(new_words)
            
            # Adiciona apenas se for uma frase nova e diferente da original
            if new_text not in seen_variants:
                seen_variants.add(new_text)
                aug_list.append(new_text)
                
        # Fallback caso o loop tente esgotar e falhe em achar únicas: preenche com o original alterado
        while len(aug_list) < num_aug:
            aug_list.append(text)
            
        augmented_results.append(aug_list)
        
    return augmented_results