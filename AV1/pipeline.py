"""Roda os scripts da AV1 em ordem; para na primeira etapa que falhar.

  python AV1/pipeline.py              # todas as etapas
  python AV1/pipeline.py --desde 2    # pula o pré-processamento
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

AV1 = Path(__file__).resolve().parent  # os scripts usam caminhos relativos a esta pasta
STEPS = {
    "1": "1_preprocessing.py",
    "2": "2_eda.py",
    "2.1": "2.1_eda_with_models.py",
    "3": "3_model_selection.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--desde", choices=STEPS, default="1", help="etapa inicial (padrão: 1)")
    args = parser.parse_args()

    order = list(STEPS)
    total = time.perf_counter()
    for step in order[order.index(args.desde):]:
        script = STEPS[step]
        print(f"\n===== [{step}] {script} =====", flush=True)
        start = time.perf_counter()
        returncode = subprocess.run([sys.executable, script], cwd=AV1).returncode
        minutes = (time.perf_counter() - start) / 60
        if returncode != 0:
            sys.exit(f"\n[{step}] {script} falhou (código {returncode}) após {minutes:.1f} min")
        print(f"[{step}] {script} concluído em {minutes:.1f} min", flush=True)
    print(f"\nPipeline concluído em {(time.perf_counter() - total) / 60:.1f} min")


if __name__ == "__main__":
    main()
