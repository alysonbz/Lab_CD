import ____
from ____
# Prepare data
rate_cuts = [0.0025, 0.001, -0.0005, -0.001, -0.0005, 0.0025, -0.001, -0.0015, -0.001, 0.0005]

# Use the whiten() function to standardize the data
scaled_data = ____(____)

# Plot original data
plt.____(____, label='original')

# Plot scaled data
plt.____(____, label='scaled')

plt.legend()
plt.show()