from scipy.ndimage import gaussian_filter
# Import Gaussian filter
from skimage.filters import gaussian
from src.pdi_utils import load_building_image,show_image
building_image = load_building_image()


# Apply filter sigma = 1
gaussian_image = gaussian(building_image, sigma=1, channel_axis=-1)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian with sigma = 1")


# Apply gaussian filter sigma = 5
gaussian_image =gaussian(building_image, sigma=5, channel_axis=-1)

# Show resulting image to compare
show_image(gaussian_image, "Reduced sharpness Gaussian with sigma = 5")

# Apply filter sigma = 10
gaussian_image = gaussian(building_image, sigma=10, channel_axis=-1)

# Show resulting image to compare
show_image(gaussian_image, "Reduced sharpness Gaussian with sigma = 10")