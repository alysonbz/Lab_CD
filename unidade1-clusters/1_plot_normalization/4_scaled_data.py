import matplotlib.pyplot as plt
# Import the whiten function
from scipy.cluster.vq import ____

goals_for = [4,3,2,3,1,1,2,0,1,4]

# Use the whiten() function to standardize the data
scaled_data = ____(____)
print(scaled_data)

# Plot original data
plt.____(____, label='original')

# Plot scaled data
plt.____(____, label='scaled')

# Show the legend in the plot
plt.____()

# Display the plot
plt.____()
