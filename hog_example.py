import skimage.io
from skimage import data, feature
import matplotlib.pyplot as plt

# fetch an image of an astronaut
image = data.astronaut()

hog_vector, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                    block_norm='L2', visualize=True)
print(hog_vector.shape)

figure, axes = plt.subplot(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
axes[0].imshow()
axes[0].set_title('Original Image')

# Show the HOG related image
axes[1].imshow(hog_image)
axes[1].set_title("HOG Image")