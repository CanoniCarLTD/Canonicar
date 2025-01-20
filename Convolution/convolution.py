import torch
import torch.nn.functional as F
import cv2
import numpy as np

# Load an RGB image using OpenCV (change the path to your image)
image_path = 'C:/Canonicar/Convolution/carla_rgb_before_conv.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: The image file '{image_path}' was not found.")

# Function to crop the center of the image to a smaller Field of View (FoV)
def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

# Set the desired cropped size (smaller FoV)
crop_width = 300  # Increased width
crop_height = 150  # Decreased height

image_cropped = crop_center(image, crop_width, crop_height)

# Convert grayscale image to a PyTorch tensor and normalize to [0,1]
image_tensor = torch.tensor(image_cropped, dtype=torch.float32) / 255.0
image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Change to (1, 1, H, W)

# Define a larger blurring kernel for a stronger blur effect
blur_kernel = torch.tensor([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
], dtype=torch.float32)

# Normalize the kernel to ensure the sum is 1 (preserving brightness)
blur_kernel = blur_kernel / blur_kernel.sum()
blur_kernel = blur_kernel.unsqueeze(0).unsqueeze(0)  # Add dimensions for convolution

# Apply the convolution using F.conv2d with appropriate padding
blurred_image = F.conv2d(image_tensor, blur_kernel, padding=3)

# Convert the blurred image back to NumPy for saving
blurred_image = blurred_image.squeeze(0).squeeze(0).detach().numpy()

# Convert from float32 [0,1] to uint8 [0,255]
blurred_image = (blurred_image * 255).astype(np.uint8)

# Save the processed image as a .jpg file
output_path = 'C:/Canonicar/Convolution/blurred_output.jpg'
cv2.imwrite(output_path, blurred_image)

print(f"Blurred image saved as '{output_path}'")
