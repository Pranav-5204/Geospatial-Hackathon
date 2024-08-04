from google.colab.patches import cv2_imshow
import cv2
import numpy as np

# Load the image
image = cv2.imread('/content/HACKATHON/preprocessed_vit3 (1).jpg')
images = []
images.append((image))

# Check if the image is 3-channels
if len(image.shape) != 3:
    raise ValueError("Image must be 3-channels")

# Create a copy of the original image
image_copy = image.copy()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny edge detector
edges = cv2.Canny(blur, 60, 150)

# Perform dilation and erosion to close gaps in between edge lines
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)
closed_edges = cv2.erode(dilated_edges, kernel, iterations=1)
# Fill in the variables
image_height = 800  # Specify the height of the images
image_width = 1700  # Specify the width of the images
num_channels = 3  # Specify the number of channels in the images (e.g., 3 for RGB)
encoding_dim = 64  # Specify the dimension of the encoded representation
num_epochs = 10  # Specify the number of epochs for training
batch_size = 32  # Specify the batch size for training
threshold_value = 0.5  # Specify the threshold value for highlighting roads

def load_preprocessed_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(directory, filename))
            # Preprocess the image as needed (e.g., resize, normalize, etc.)
            # Append the preprocessed image to the list
            images.append((image))
    return np.array(images)

