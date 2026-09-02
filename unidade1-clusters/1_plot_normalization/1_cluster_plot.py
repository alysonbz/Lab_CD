from src.utils import load_pokemon_dataset

x,y = load_pokemon_dataset()

# Import plotting class from matplotlib library
from matplotlib import pyplot as plt

# Create a scatter plot
plt.scatter(x, y)

# Display the scatter plot
plt.show()
