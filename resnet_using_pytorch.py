import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights, ResNet18_Weights, VGG16_Weights, DenseNet121_Weights, EfficientNet_B0_Weights


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet values)
])

# Load datasets
train_dataset = datasets.ImageFolder(root="train", transform=transform)
val_dataset = datasets.ImageFolder(root="val", transform=transform)
test_dataset = datasets.ImageFolder(root="test", transform=transform)

# Create DataLoaders for batching and shuffling
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Extract class labels
class_names = train_dataset.classes  # List of class names
print(f"Classes: {class_names}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create DataLoaders from preprocessed datasets
def create_dataloader(X, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Define batch size
batch_size = 32

# Pretrained models
model_dict = {
    "ResNet18": models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    "ResNet50": models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
    "VGG16": models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
    "DenseNet": models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
    "EfficientNet": models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
}

# Determine the number of classes from the train_loader
def get_num_classes(data_loader):
    unique_classes = set()
    for _, labels in data_loader:
        unique_classes.update(labels.numpy())  # Collect unique labels
    return len(unique_classes)

# Replace the final layer with the number of classes in the dataset
num_classes = get_num_classes(train_loader)

for model_name, model in model_dict.items():
    if "resnet" in model_name.lower():  # ResNet models
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif "vgg" in model_name.lower():  # VGG models
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif "densenet" in model_name.lower():  # DenseNet models
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif "efficientnet" in model_name.lower():  # EfficientNet models
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)


# Training function with logging
def train_model_with_logging(model, train_loader, val_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0

    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_losses.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        val_acc = evaluate_model(model, val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    # Load best model weights
    model.load_state_dict(best_model)
    training_time = time.time() - start_time

    return model, train_losses, val_losses, best_val_acc, training_time

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total

# Train and evaluate all models
results = {}
train_times = []
all_train_losses = {}
all_val_losses = {}

epochs = 10
for model_name, model in model_dict.items():
    print(f"\nTraining {model_name}...")
    model, train_losses, val_losses, val_acc, training_time = train_model_with_logging(
        model, train_loader, val_loader, epochs=epochs
    )
    test_acc = evaluate_model(model, test_loader)
    print(f"Test Accuracy for {model_name}: {test_acc:.2f}%")
    
    # Store results
    results[model_name] = {"Test Accuracy": test_acc, "Validation Accuracy": val_acc}
    train_times.append(training_time)
    all_train_losses[model_name] = train_losses
    all_val_losses[model_name] = val_losses

# Plot Training and Validation Losses
plt.figure(figsize=(12, 6))
for model_name in model_dict.keys():
    plt.plot(all_train_losses[model_name], label=f"{model_name} - Train Loss")
    plt.plot(all_val_losses[model_name], linestyle="--", label=f"{model_name} - Val Loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot Training Time Comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), train_times, color='skyblue')
plt.title("Training Time Comparison")
plt.xlabel("Models")
plt.ylabel("Training Time (seconds)")
plt.show()

# Plot Test Accuracy Comparison
test_accuracies = [results[model]["Test Accuracy"] for model in results]
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), test_accuracies, color='lightgreen')
plt.title("Test Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.show()

# Final results summary
print("\nFinal Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: Test Accuracy = {metrics['Test Accuracy']:.2f}%, Validation Accuracy = {metrics['Validation Accuracy']:.2f}%")