import sys
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.pdi_utils import load_lena
import matplotlib.pyplot as plt

from src.pdi_utils import load_red_roses,show_image
import matplotlib.pyplot as plt

image = load_red_roses()

# Show original image
show_image(image,'image RGB')

# Obtain the red channel
red_channel = image[:, :, 1]

# Show red image
show_image(red_channel,'image red channel')

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel, bins=256)

# Set title and show
plt.title('Red Histogram')
plt.show()