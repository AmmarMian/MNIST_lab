"""
Part 3: Transfer Learning with Fashion-MNIST Complete Solution
Transfer Learning using Pre-trained Models on Fashion-MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns


# ============================================================================
# Configuration
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fashion-MNIST class names
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# Data Loading for Fashion-MNIST
# ============================================================================


def load_fashion_mnist(batch_size=64):
    """Load Fashion-MNIST dataset with appropriate transforms"""

    # Define transforms
    # Pre-trained models expect normalized inputs
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    # Load Fashion-MNIST datasets
    train_dataset = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader


# ============================================================================
# Transfer Learning Model
# ============================================================================


class TransferLearningModel(nn.Module):
    """Transfer learning model for Fashion-MNIST"""

    def __init__(self, num_classes=10, feature_extract=True, model_name="resnet18"):
        super(TransferLearningModel, self).__init__()

        self.model_name = model_name

        # Load pre-trained model
        if model_name == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
            num_features = self.base_model.fc.in_features
            # Replace final layer
            self.base_model.fc = nn.Linear(num_features, num_classes)

        elif model_name == "mobilenet_v2":
            self.base_model = models.mobilenet_v2(pretrained=True)
            # MobileNetV2 has a different structure
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_features, num_classes)

        elif model_name == "efficientnet_b0":
            # Note: EfficientNet might require separate installation
            # Using a simpler alternative
            self.base_model = models.resnet18(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_classes)
            print("Note: Using ResNet18 as EfficientNet substitute")

        # Freeze layers if feature_extract=True
        if feature_extract:
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Unfreeze the final layer
            if model_name == "resnet18":
                for param in self.base_model.fc.parameters():
                    param.requires_grad = True
            elif model_name == "mobilenet_v2":
                for param in self.base_model.classifier[1].parameters():
                    param.requires_grad = True

        # Add layer to handle grayscale to RGB conversion
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

    def forward(self, x):
        """Forward pass"""
        # Convert grayscale to RGB
        x = self.gray_to_rgb(x)
        # Pass through base model
        x = self.base_model(x)
        return x

    def unfreeze_layers(self, num_layers_to_unfreeze):
        """Unfreeze the last N layers for fine-tuning"""
        if self.model_name == "resnet18":
            # ResNet18 has layer1, layer2, layer3, layer4
            layers = [
                self.base_model.layer4,
                self.base_model.layer3,
                self.base_model.layer2,
                self.base_model.layer1,
            ]

            for i in range(min(num_layers_to_unfreeze, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = True

        elif self.model_name == "mobilenet_v2":
            # MobileNetV2 has features with multiple blocks
            features = list(self.base_model.features.children())
            num_unfrozen = 0

            # Unfreeze from the end
            for i in range(len(features) - 1, -1, -1):
                if num_unfrozen >= num_layers_to_unfreeze:
                    break
                for param in features[i].parameters():
                    param.requires_grad = True
                num_unfrozen += 1

        # Always ensure final layer is trainable
        if self.model_name == "resnet18":
            for param in self.base_model.fc.parameters():
                param.requires_grad = True
        elif self.model_name == "mobilenet_v2":
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )


# ============================================================================
# Training Functions
# ============================================================================


def train_transfer_model(
    model, train_loader, test_loader, epochs=10, lr=0.001, fine_tune=False
):
    """
    Train transfer learning model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer that only updates parameters with requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Add scheduler for fine-tuning
    scheduler = None
    if fine_tune:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

    history = {"train_losses": [], "train_accs": [], "test_losses": [], "test_accs": []}

    print(
        f"\nTraining {'with fine-tuning' if fine_tune else 'feature extraction only'}..."
    )
    print(f"Learning rate: {lr}")
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

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}",
                    end="\r",
                )

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)

        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        history["test_losses"].append(test_loss)
        history["test_accs"].append(test_acc)

        # Update scheduler if fine-tuning
        if scheduler:
            scheduler.step(test_acc)

        # Print progress
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, LR: {current_lr:.6f}"
        )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final test accuracy: {history['test_accs'][-1]:.4f}")

    return history


def two_stage_training(model_name="resnet18"):
    """
    Implement two-stage training strategy:
    Stage 1: Feature extraction (frozen backbone)
    Stage 2: Fine-tuning (unfrozen layers)
    """

    # Load data
    train_loader, test_loader = load_fashion_mnist()

    # Stage 1: Feature extraction
    print("\n" + "=" * 60)
    print("Stage 1: Feature Extraction (Frozen Backbone)")
    print("=" * 60)

    # Create model with feature_extract=True
    model = TransferLearningModel(
        num_classes=10, feature_extract=True, model_name=model_name
    )

    # Train with frozen backbone
    # Use higher learning rate since we're only training the final layer
    stage1_history = train_transfer_model(
        model, train_loader, test_loader, epochs=5, lr=0.001, fine_tune=False
    )

    # Stage 2: Fine-tuning
    print("\n" + "=" * 60)
    print("Stage 2: Fine-tuning (Unfreezing Layers)")
    print("=" * 60)

    # Unfreeze some layers
    print("\nUnfreezing last 2 layer blocks...")
    model.unfreeze_layers(2)

    # Train with lower learning rate
    stage2_history = train_transfer_model(
        model, train_loader, test_loader, epochs=10, lr=0.0001, fine_tune=True
    )

    # Combine histories
    combined_history = {
        "train_losses": stage1_history["train_losses"] + stage2_history["train_losses"],
        "train_accs": stage1_history["train_accs"] + stage2_history["train_accs"],
        "test_losses": stage1_history["test_losses"] + stage2_history["test_losses"],
        "test_accs": stage1_history["test_accs"] + stage2_history["test_accs"],
    }

    return model, stage1_history, stage2_history, combined_history


# ============================================================================
# Comparison Functions
# ============================================================================


def train_from_scratch_cnn():
    """Train a CNN from scratch on Fashion-MNIST for comparison"""

    class FashionCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(FashionCNN, self).__init__()
            # Similar architecture to Part 2
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Load data
    train_loader, test_loader = load_fashion_mnist()

    # Create and train model
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_losses": [], "train_accs": [], "test_losses": [], "test_accs": []}

    print("\nTraining CNN from scratch...")
    start_time = time.time()

    epochs = 15
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)

        # Testing
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_acc = correct / total
        history["test_losses"].append(test_loss)
        history["test_accs"].append(test_acc)

        print(f"Epoch {epoch}: Test Acc: {test_acc:.4f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    return model, history


def compare_approaches():
    """Compare transfer learning vs training from scratch"""

    print("\nComparing Transfer Learning vs Training from Scratch")
    print("=" * 60)

    # Train from scratch model
    print("\n1. Training from Scratch")
    print("-" * 40)
    scratch_model, scratch_history = train_from_scratch_cnn()

    # Train transfer learning model
    print("\n2. Transfer Learning")
    print("-" * 40)
    transfer_model, stage1_hist, stage2_hist, combined_hist = two_stage_training()

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test accuracy comparison
    axes[0, 0].plot(scratch_history["test_accs"], label="From Scratch", linewidth=2)
    axes[0, 0].plot(combined_hist["test_accs"], label="Transfer Learning", linewidth=2)
    axes[0, 0].axvline(
        x=len(stage1_hist["test_accs"]) - 1,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Fine-tuning starts",
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Test Accuracy")
    axes[0, 0].set_title("Test Accuracy Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test loss comparison
    axes[0, 1].plot(scratch_history["test_losses"], label="From Scratch", linewidth=2)
    axes[0, 1].plot(
        combined_hist["test_losses"], label="Transfer Learning", linewidth=2
    )
    axes[0, 1].axvline(
        x=len(stage1_hist["test_losses"]) - 1, color="red", linestyle="--", alpha=0.5
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Test Loss")
    axes[0, 1].set_title("Test Loss Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Final accuracy bar plot
    final_accs = [scratch_history["test_accs"][-1], combined_hist["test_accs"][-1]]
    methods = ["From Scratch", "Transfer Learning"]
    bars = axes[1, 0].bar(methods, final_accs, color=["blue", "green"])
    axes[1, 0].set_ylabel("Final Test Accuracy")
    axes[1, 0].set_title("Final Model Performance")
    axes[1, 0].set_ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )

    # Training efficiency (epochs to reach 85% accuracy)
    target_acc = 0.85
    epochs_scratch = next(
        (i for i, acc in enumerate(scratch_history["test_accs"]) if acc >= target_acc),
        len(scratch_history["test_accs"]),
    )
    epochs_transfer = next(
        (i for i, acc in enumerate(combined_hist["test_accs"]) if acc >= target_acc),
        len(combined_hist["test_accs"]),
    )

    efficiency_data = [epochs_scratch, epochs_transfer]
    bars = axes[1, 1].bar(methods, efficiency_data, color=["blue", "green"])
    axes[1, 1].set_ylabel("Epochs to 85% Accuracy")
    axes[1, 1].set_title("Training Efficiency")

    # Add value labels
    for bar, epochs in zip(bars, efficiency_data):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{epochs}",
            ha="center",
            va="bottom",
        )

    plt.suptitle("Transfer Learning vs Training from Scratch", fontsize=16)
    plt.tight_layout()
    plt.show()

    return scratch_model, transfer_model, scratch_history, combined_hist


# ============================================================================
# Visualization Functions
# ============================================================================


def visualize_predictions(model, test_loader, num_images=10):
    """Visualize model predictions on test images"""
    model.eval()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(device), labels[:num_images]

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

    # Move to CPU for plotting
    images = images.cpu()
    predicted = predicted.cpu()
    probs = probs.cpu()

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for idx in range(num_images):
        # Show image
        img = images[idx].squeeze()
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        axes[idx].imshow(img, cmap="gray")

        # Add prediction info
        true_label = classes[labels[idx]]
        pred_label = classes[predicted[idx]]
        confidence = probs[idx, predicted[idx]].item()

        color = "green" if predicted[idx] == labels[idx] else "red"
        axes[idx].set_title(
            f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}",
            color=color,
            fontsize=10,
        )
        axes[idx].axis("off")

    plt.suptitle("Model Predictions on Fashion-MNIST", fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_feature_representations(model, loader, num_samples=1000):
    """Analyze what features the model learned using t-SNE"""

    model.eval()
    features = []
    labels = []

    # Extract features from the second-to-last layer
    def get_features(module, input, output):
        features.append(input[0].cpu().numpy())

    # Register hook
    if hasattr(model, "base_model"):
        if hasattr(model.base_model, "fc"):
            handle = model.base_model.fc.register_forward_hook(get_features)
        else:
            handle = model.base_model.classifier[-1].register_forward_hook(get_features)

    # Extract features
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if len(labels) >= num_samples:
                break
            data = data.to(device)
            _ = model(data)
            labels.extend(target.numpy())

    # Remove hook
    handle.remove()

    # Convert to numpy arrays
    features = np.vstack(features[:num_samples])
    labels = np.array(labels[:num_samples])

    # Apply t-SNE
    print("Applying t-SNE to feature representations...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab10", alpha=0.6
    )
    plt.colorbar(scatter, ticks=range(10))

    # Add class labels
    for i in range(10):
        class_points = features_2d[labels == i]
        if len(class_points) > 0:
            center = class_points.mean(axis=0)
            plt.annotate(
                classes[i],
                center,
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    plt.title("t-SNE Visualization of Learned Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.show()


def evaluate_detailed(model, test_loader):
    """Detailed evaluation with confusion matrix"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_targets, all_predictions, target_names=classes, digits=3
        )
    )

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Calculate per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 40)
    for i in range(10):
        class_mask = all_targets == i
        if class_mask.sum() > 0:
            class_acc = (all_predictions[class_mask] == i).mean()
            print(f"{classes[i]:15s}: {class_acc:.3f}")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main execution function"""

    print("Fashion-MNIST Transfer Learning Lab")
    print("=" * 60)

    # Load and visualize data
    train_loader, test_loader = load_fashion_mnist()

    # Visualize samples
    print("\nVisualizing Fashion-MNIST samples...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i, (img, label) in enumerate(train_loader):
        if i >= 10:
            break
        img_show = img[0].squeeze().numpy()
        # Denormalize
        img_show = (img_show + 1) / 2
        axes[i].imshow(img_show, cmap="gray")
        axes[i].set_title(f"{classes[label[0]]}")
        axes[i].axis("off")

    plt.suptitle("Fashion-MNIST Samples")
    plt.tight_layout()
    plt.show()

    # ========================================
    # Task 1: Basic Transfer Learning
    # ========================================
    print("\n" + "=" * 60)
    print("Task 1: Basic Transfer Learning")
    print("=" * 60)

    # Create and train transfer learning model
    model = TransferLearningModel(
        num_classes=10, feature_extract=True, model_name="resnet18"
    )

    history = train_transfer_model(model, train_loader, test_loader, epochs=5, lr=0.001)

    # Visualize predictions
    visualize_predictions(model, test_loader)

    # ========================================
    # Task 2: Two-Stage Training
    # ========================================
    print("\n" + "=" * 60)
    print("Task 2: Two-Stage Training")
    print("=" * 60)

    transfer_model, stage1_hist, stage2_hist, combined_hist = two_stage_training()

    # Plot two-stage training progress
    plt.figure(figsize=(10, 6))
    epochs_stage1 = len(stage1_hist["test_accs"])
    total_epochs = len(combined_hist["test_accs"])

    plt.plot(
        range(epochs_stage1),
        stage1_hist["test_accs"],
        "bo-",
        label="Stage 1: Feature Extraction",
        linewidth=2,
    )
    plt.plot(
        range(epochs_stage1, total_epochs),
        stage2_hist["test_accs"],
        "ro-",
        label="Stage 2: Fine-tuning",
        linewidth=2,
    )
    plt.axvline(x=epochs_stage1 - 0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Two-Stage Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ========================================
    # Task 3: Comparison Study
    # ========================================
    print("\n" + "=" * 60)
    print("Task 3: Comparison with Training from Scratch")
    print("=" * 60)

    scratch_model, transfer_model, scratch_hist, transfer_hist = compare_approaches()

    # ========================================
    # Task 4: Model Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("Task 4: Analyzing Learned Representations")
    print("=" * 60)

    # Analyze feature representations
    print("\nAnalyzing transfer learning model features...")
    analyze_feature_representations(transfer_model, test_loader)

    # Detailed evaluation
    print("\nDetailed evaluation of transfer learning model:")
    evaluate_detailed(transfer_model, test_loader)

    # Compare model sizes and efficiency
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)

    # Count parameters
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    scratch_total, scratch_trainable = count_parameters(scratch_model)
    transfer_total, transfer_trainable = count_parameters(transfer_model)

    print(f"From Scratch Model:")
    print(f"  Total parameters: {scratch_total:,}")
    print(f"  Trainable parameters: {scratch_trainable:,}")

    print(f"\nTransfer Learning Model:")
    print(f"  Total parameters: {transfer_total:,}")
    print(f"  Trainable parameters: {transfer_trainable:,}")

    print(f"\nFinal Test Accuracies:")
    print(f"  From Scratch: {scratch_hist['test_accs'][-1]:.4f}")
    print(f"  Transfer Learning: {transfer_hist['test_accs'][-1]:.4f}")

    return transfer_model


if __name__ == "__main__":
    # Run the complete solution
    final_model = main()

    print("\n" + "=" * 60)
    print("Part 3 Complete!")
    print("=" * 60)

    # Save the best model
    torch.save(final_model.state_dict(), "fashion_mnist_transfer.pth")
    print("\nBest model saved as 'fashion_mnist_transfer.pth'")
