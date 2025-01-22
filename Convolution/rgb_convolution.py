import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the convolutional network


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        # Adjust input channels to 1 for grayscale images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # Conv Layer 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # Conv Layer 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)  # Conv Layer 3 (4 channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 6))  # Ensure the output size is (5, 6)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten starting from the second dimension
        return x


# Initialize the convolutional model
conv_model = DeepConvNet()

# Function to preprocess the input image


def preprocess_image(image: np.ndarray, new_height: int = 256) -> torch.Tensor:
    """
    Preprocess the input image: resize, normalize, and convert to tensor.

    Args:
        image (np.ndarray): Input image as a NumPy array (H x W).
        new_height (int): Desired height for resizing (default: 256).

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    width, height = image.shape
    new_width = int(new_height * (width / height))  # Maintain aspect ratio
    image_pil = Image.fromarray(image)  # Convert to PIL image

    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjusted for grayscale
    ])

    return transform(image_pil).unsqueeze(0)  # Add batch dimension

# Function to extract features from the image using the convolutional model


def extract_features(image_tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Extract features from the input image tensor using the convolutional model.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor (1 x C x H x W).
        model (nn.Module): Convolutional model for feature extraction.

    Returns:
        torch.Tensor: Flattened feature vector.
    """
    start = time.time()
    with torch.no_grad():
        output = model(image_tensor)
    end = time.time()
    print(f"Feature extraction time: {end - start:.4f} seconds")
    return output


# Example workflow
if __name__ == "__main__":
    # Simulate a random CARLA image (800x600)
    carla_image_path = "Convolution/000171.png"

    image = np.array(Image.open(carla_image_path).convert('L'))
    print(image.shape)
    # Preprocess the image
    preprocessed_image = preprocess_image(carla_image_path)

    # Extract features
    features = extract_features(preprocessed_image, conv_model)

    # Verify the feature vector shape
    print("Flattened output shape:", features.shape)
    assert features.shape[1] == 120, "The flattened output does not have 120 elements!"
