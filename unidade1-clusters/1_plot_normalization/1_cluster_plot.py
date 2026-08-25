import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils import load_pokemon_dataset

x,y = load_pokemon_dataset()

from matplotlib import pyplot as plt

plt.scatter(x, y)

plt.show()