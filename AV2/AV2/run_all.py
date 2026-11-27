"""
run_all.py - executa o projeto completo (Passos 1 a 6) em sequência.

    python run_all.py              # tudo, retomando do cache quando existir
    python run_all.py --from 4     # recomeça a partir do passo 4
    python run_all.py --force      # refaz varreduras/treinos mesmo com cache

Cada passo também pode ser executado isoladamente (python stepXX_*.py).
"""
import argparse
import time

import step01_dataset_analysis as s1
import step02_preprocessing as s2
import step03_segmentation as s3
import step04_feature_extraction as s4
import step05_train_evaluate as s5
import step06_analysis_report as s6
from utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="passo inicial (1-6)")
    ap.add_argument("--force", action="store_true", help="ignora caches")
    ap.add_argument("--rapido", action="store_true", help="Passo 5 em modo rápido (RF mais leve; ~6x mais rápido)")
    a = ap.parse_args()
    t0 = time.time()
    steps = [
        (1, "Análise e preparação do dataset", lambda: s1.main(a.force)),
        (2, "Pré-processamento", lambda: s2.main(a.force)),
        (3, "Segmentação", lambda: s3.main()),
        (4, "Extração de características", lambda: s4.main()),
        (5, "Treinamento e avaliação", lambda: s5.main(None, a.force, a.rapido)),
        (6, "Análise e relatório", lambda: s6.main()),
    ]
    for n, name, fn in steps:
        if n < a.start:
            continue
        log(f"================ PASSO {n}: {name} ================")
        fn()
    log(f"PROJETO CONCLUÍDO em {(time.time() - t0) / 60:.1f} min. Veja a pasta outputs/.")


if __name__ == "__main__":
    main()
