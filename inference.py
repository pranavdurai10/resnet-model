'''
/////////////////////////////////////////////////////
Code written by Pranav Durai on 24.05.2023 @ 21:11:17

ResNet Inference Script

Framework: PyTorch 2.0
/////////////////////////////////////////////////////
'''

# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.resnet18 import ResNet18
from models.resnet34 import ResNet34
from PIL import Image


# Load the pretrained ResNet-18 model
model = ResNet18()
model.load_state_dict(torch.load('resnet18.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load and preprocess the image
image_path = 'PATH_TO_IMAGE_DIRECTORY'
image = Image.open(image_path)
image = transform(image).unsqueeze(0)

# Perform the inference
with torch.no_grad():
    output = model(image)

# Get the predicted class label
_, predicted = torch.max(output, 1)
predicted_label = predicted.item()

# Load the class labels
with open('classes.txt', 'r') as f:
    classes = f.readlines()

# Print the predicted class label
print('Predicted label:', classes[predicted_label])