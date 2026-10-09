import numpy as np

def worker_eda_chunk(args: tuple) -> list:
    """Worker para aumento de dados textuais via EDA."""
    chunk_texts, num_aug = args
    augmented_results = []
    for text in chunk_texts:
        words = text.split()
        if not words or len(words) <= 2:
            augmented_results.append([text] * num_aug)
            continue
        aug_list = []
        for _ in range(num_aug):
            new_words = words.copy()
            idx1, idx2 = np.random.choice(len(new_words), 2, replace=False)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            aug_list.append(" ".join(new_words))
        augmented_results.append(aug_list)
    return augmented_results