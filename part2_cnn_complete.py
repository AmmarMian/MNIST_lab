"""
Part 2: Deep Learning with CNNs Complete Solution
MNIST Classification using Convolutional Neural Networks in PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import seaborn as sns


# ============================================================================
# Configuration and Setup
# ============================================================================

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================


def load_mnist_data():
    """Load MNIST data and convert to PyTorch tensors"""
    print("Loading MNIST dataset...")

    # Load from OpenML
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    # Convert to appropriate types
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Create train/val/test splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Convert to PyTorch tensors and reshape for CNN (N, C, H, W)
    X_train_tensor = torch.FloatTensor(X_train.reshape(-1, 1, 28, 28)) / 255.0
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val.reshape(-1, 1, 28, 28)) / 255.0
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test.reshape(-1, 1, 28, 28)) / 255.0
    y_test_tensor = torch.LongTensor(y_test)

    print(f"Training set: {X_train_tensor.shape}")
    print(f"Validation set: {X_val_tensor.shape}")
    print(f"Test set: {X_test_tensor.shape}")

    return (
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        X_test_tensor,
        y_test_tensor,
    )


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """Create PyTorch DataLoaders"""
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")

    return train_loader, val_loader, test_loader


# ============================================================================
# CNN Architecture
# ============================================================================


class SimpleCNN(nn.Module):
    """Simple CNN architecture for MNIST classification"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_feature_maps(self, x):
        """Extract feature maps for visualization"""
        features = {}

        # First conv block
        x = self.conv1(x)
        features["conv1"] = x.detach().cpu()
        x = self.bn1(x)
        x = F.relu(x)
        features["conv1_activated"] = x.detach().cpu()
        x = self.pool(x)

        # Second conv block
        x = self.conv2(x)
        features["conv2"] = x.detach().cpu()
        x = self.bn2(x)
        x = F.relu(x)
        features["conv2_activated"] = x.detach().cpu()

        return features


class ImprovedCNN(nn.Module):
    """Improved CNN with more layers and advanced features"""

    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()

        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# Training Functions
# ============================================================================


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    learning_rate=0.001,
    optimizer_name="Adam",
    scheduler_type=None,
):
    """
    Train the model with specified optimizer and optional scheduler
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Select scheduler
    scheduler = None
    if scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

    # Training history
    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "learning_rates": [],
    }

    print(
        f"\nTraining with {optimizer_name} optimizer"
        + (f" and {scheduler_type} scheduler" if scheduler else "")
    )
    print("=" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Print progress
            if batch_idx % 200 == 0:
                print(
                    f"Epoch: {epoch:3d}, Batch: {batch_idx:4d}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}",
                    end="\r",
                )

        # Calculate epoch statistics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        if scheduler:
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Print epoch summary
        print(
            f"\nEpoch: {epoch:3d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")

    return history


# ============================================================================
# Evaluation and Visualization Functions
# ============================================================================


def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(history["train_losses"], label="Train Loss", linewidth=2)
    axes[0, 0].plot(history["val_losses"], label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(history["train_accuracies"], label="Train Acc", linewidth=2)
    axes[0, 1].plot(history["val_accuracies"], label="Val Acc", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(history["learning_rates"], linewidth=2, color="green")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # Train vs Val accuracy gap
    accuracy_gap = np.array(history["train_accuracies"]) - np.array(
        history["val_accuracies"]
    )
    axes[1, 1].plot(accuracy_gap, linewidth=2, color="orange")
    axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Train - Val Accuracy")
    axes[1, 1].set_title("Generalization Gap")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, class_names=None):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    # Get predictions
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = np.mean(all_predictions == all_targets)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")

    # Per-class accuracy
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    print("\nPer-class Performance:")
    print("-" * 40)
    for i in range(10):
        class_mask = all_targets == i
        class_acc = np.mean(all_predictions[class_mask] == i)
        class_count = np.sum(class_mask)
        print(
            f"Class {i} ({class_names[i]:>10}): {class_acc:.4f} ({class_count} samples)"
        )

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return accuracy, all_predictions, all_probs


def visualize_filters_and_features(model, sample_loader):
    """Visualize CNN filters and feature maps"""
    # Get a sample batch
    data, _ = next(iter(sample_loader))
    sample_image = data[0:1].to(device)

    # 1. Visualize first layer filters
    if hasattr(model, "conv1"):
        conv1_weights = model.conv1.weight.data.cpu()
    else:
        conv1_weights = model.conv_block1[0].weight.data.cpu()

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(32, conv1_weights.shape[0])):
        filter_i = conv1_weights[i, 0, :, :]
        axes[i].imshow(filter_i, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"F{i}", fontsize=8)

    plt.suptitle("First Layer Convolutional Filters")
    plt.tight_layout()
    plt.show()

    # 2. Visualize feature maps
    if hasattr(model, "get_feature_maps"):
        model.eval()
        features = model.get_feature_maps(sample_image)

        # Plot conv1 features
        conv1_features = features["conv1_activated"].squeeze(0)

        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        axes = axes.ravel()

        for i in range(min(32, conv1_features.shape[0])):
            axes[i].imshow(conv1_features[i], cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"F{i}", fontsize=8)

        plt.suptitle("Feature Maps from First Convolutional Layer")
        plt.tight_layout()
        plt.show()


def compare_optimizers(model_class, train_loader, val_loader, epochs=15):
    """Compare different optimizers"""
    optimizers = ["SGD", "Adam", "RMSprop"]
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for opt_name in optimizers:
        print(f"\n{'=' * 60}")
        print(f"Training with {opt_name} optimizer")
        print("=" * 60)

        # Create fresh model
        model = model_class()

        # Train
        history = train_model(
            model, train_loader, val_loader, epochs=epochs, optimizer_name=opt_name
        )

        results[opt_name] = history

        # Plot results
        axes[0].plot(history["val_losses"], label=opt_name, linewidth=2)
        axes[1].plot(history["val_accuracies"], label=opt_name, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Validation Accuracy Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Optimizer Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

    return results


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main execution function"""

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    # Visualize some samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i, (img, label) in enumerate(train_loader):
        if i >= 10:
            break
        img_show = img[0].squeeze().numpy()
        axes[i].imshow(img_show, cmap="gray")
        axes[i].set_title(f"Label: {label[0].item()}")
        axes[i].axis("off")

    plt.suptitle("Sample MNIST Digits")
    plt.tight_layout()
    plt.show()

    # ========================================
    # Experiment 1: Basic CNN Training
    # ========================================
    print("\n" + "=" * 60)
    print("Experiment 1: Basic CNN Training")
    print("=" * 60)

    model = SimpleCNN()
    history = train_model(model, train_loader, val_loader, epochs=20)
    plot_training_history(history, "Basic CNN Training")

    # Evaluate on test set
    test_accuracy, predictions, probs = evaluate_model(model, test_loader)

    # Visualize filters and features
    visualize_filters_and_features(model, val_loader)

    # ========================================
    # Experiment 2: Optimizer Comparison
    # ========================================
    print("\n" + "=" * 60)
    print("Experiment 2: Comparing Optimizers")
    print("=" * 60)

    optimizer_results = compare_optimizers(SimpleCNN, train_loader, val_loader)

    # ========================================
    # Experiment 3: Learning Rate Scheduling
    # ========================================
    print("\n" + "=" * 60)
    print("Experiment 3: Learning Rate Scheduling")
    print("=" * 60)

    # Train with ReduceLROnPlateau
    model_scheduled = SimpleCNN()
    history_scheduled = train_model(
        model_scheduled,
        train_loader,
        val_loader,
        epochs=30,
        scheduler_type="ReduceLROnPlateau",
    )
    plot_training_history(history_scheduled, "Training with ReduceLROnPlateau")

    # ========================================
    # Experiment 4: Improved Architecture
    # ========================================
    print("\n" + "=" * 60)
    print("Experiment 4: Improved CNN Architecture")
    print("=" * 60)

    improved_model = ImprovedCNN()
    history_improved = train_model(
        improved_model, train_loader, val_loader, epochs=20, learning_rate=0.001
    )
    plot_training_history(history_improved, "Improved CNN Training")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Model Comparison")
    print("=" * 60)

    print("Basic CNN:")
    basic_acc, _, _ = evaluate_model(model, test_loader)

    print("\nImproved CNN:")
    improved_acc, _, _ = evaluate_model(improved_model, test_loader)

    print(f"\nAccuracy improvement: {(improved_acc - basic_acc) * 100:.2f}%")

    # Save the best model
    best_model = improved_model if improved_acc > basic_acc else model
    torch.save(best_model.state_dict(), "best_mnist_cnn.pth")
    print("\nBest model saved as 'best_mnist_cnn.pth'")

    return model, improved_model, history, history_improved


if __name__ == "__main__":
    # Run the complete solution
    basic_model, improved_model, basic_history, improved_history = main()

    print("\n" + "=" * 60)
    print("Part 2 Complete!")
    print("=" * 60)
