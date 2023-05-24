'''
/////////////////////////////////////////////////////
Code written by Pranav Durai on 24.05.2023 @ 21:11:17

ResNet Training Script

Framework: PyTorch 2.0
/////////////////////////////////////////////////////
'''

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.resnet18 import ResNet18
from models.resnet34 import ResNet34

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
valid_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

# Create an instance of ResNet model - Replace based on the model requirement
model = ResNet18().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track training loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Validation
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    # Print epoch statistics
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss / len(train_loader):.4f}  Train Accuracy: {100 * train_correct / train_total:.2f}%")
    print(f"Valid Loss: {valid_loss / len(valid_loader):.4f}  Valid Accuracy: {100 * valid_correct / valid_total:.2f}%")
    print()

# Save the trained model
torch.save(model.state_dict(), "resnet18_cifar10.pth")