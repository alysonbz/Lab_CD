import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from src.utils import load_pokemon_dataset

# Load the dataset
x, y = load_pokemon_dataset()

# Use the whiten() function to standardize the data
scaled_x = whiten(x)
scaled_y = whiten(y)

# Plot the original data in a line plot
plt.plot(x, label='x_original')
plt.plot(y, label='y_original')
plt.legend()
plt.title('Dados originais (sem normalização)')
plt.show()

# Plot the scaled (whitened) data in a line plot
plt.plot(scaled_x, label='x_scaled')
plt.plot(scaled_y, label='y_scaled')
plt.legend()
plt.title('Dados normalizados (whiten)')
plt.show()

# Show the differences between original and scaled data
plt.plot(x, label='x_original')
plt.plot(scaled_x, label='x_scaled')
plt.legend()
plt.title('Diferença: x original vs x normalizado')
plt.show()

plt.plot(y, label='y_original')
plt.plot(scaled_y, label='y_scaled')
plt.legend()
plt.title('Diferença: y original vs y normalizado')
plt.show()