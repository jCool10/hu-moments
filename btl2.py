from skimage.io import imread
from skimage.feature import hog
from skimage import exposure
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

# Read the image
img = imread("images/ot.jpg")
img = rgb2gray(img)


# Calculate HOG features
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Print the shape of image features
print('\n\nShape of Image Features\n\n')
print(fd.shape)

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

# Plot the input image
ax[0].imshow(img, cmap="gray")
ax[0].set_title('Input image')

# Plot the histogram of oriented gradients
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax[1].imshow(hog_image_rescaled , cmap ="gray")
ax[1].set_title('Histogram of Oriented Gradients')

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()
