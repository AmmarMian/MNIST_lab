"""
Utilities for MNIST and Fashion-MNIST Lab Sessions
Common functions for data loading, visualization, and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


# ============================================================================
# Visualization Utilities
# ============================================================================


def plot_samples(
    X,
    y,
    title="Sample Images",
    num_samples=10,
    figsize=(12, 5),
    class_names=None,
    reshape_size=(28, 28),
):
    """
    Plot sample images in a grid

    Args:
        X: Image data (numpy array or torch tensor)
        y: Labels
        title: Title for the plot
        num_samples: Number of samples to display
        figsize: Figure size
        class_names: List of class names (optional)
        reshape_size: Size to reshape flat images
    """
    # Convert torch tensors to numpy if needed
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()

    # Determine grid size
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel() if num_samples > 1 else [axes]

    for i in range(min(num_samples, len(X))):
        # Handle different input shapes
        if X[i].ndim == 1:
            img = X[i].reshape(reshape_size)
        elif X[i].ndim == 2:
            img = X[i]
        elif X[i].ndim == 3:
            img = X[i].squeeze()

        # Normalize to [0, 1] if needed
        if img.min() < 0:
            img = (img + 1) / 2  # From [-1, 1] to [0, 1]
        elif img.max() > 1:
            img = img / 255.0  # From [0, 255] to [0, 1]

        axes[i].imshow(img, cmap="gray")

        # Set title
        if class_names is not None:
            axes[i].set_title(f"{class_names[y[i]]}")
        else:
            axes[i].set_title(f"Label: {y[i]}")

        axes[i].axis("off")

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title="Training History", figsize=(14, 5)):
    """
    Plot training and validation curves

    Args:
        history: Dictionary with keys like 'train_losses', 'val_losses', etc.
        title: Title for the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot losses if available
    if "train_losses" in history:
        axes[0].plot(history["train_losses"], label="Train Loss", linewidth=2)
    if "val_losses" in history or "test_losses" in history:
        val_key = "val_losses" if "val_losses" in history else "test_losses"
        axes[0].plot(history[val_key], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracies if available
    if "train_accs" in history or "train_accuracies" in history:
        acc_key = "train_accs" if "train_accs" in history else "train_accuracies"
        axes[1].plot(history[acc_key], label="Train Accuracy", linewidth=2)
    if "val_accs" in history or "val_accuracies" in history or "test_accs" in history:
        val_keys = ["val_accs", "val_accuracies", "test_accs", "test_accuracies"]
        for key in val_keys:
            if key in history:
                axes[1].plot(history[key], label="Validation Accuracy", linewidth=2)
                break
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    figsize=(10, 8),
    normalize=False,
):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the plot
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return cm


# ============================================================================
# Evaluation Utilities
# ============================================================================


def evaluate_model(model, data_loader, device="cpu", return_predictions=False):
    """
    Evaluate a PyTorch model

    Args:
        model: PyTorch model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
        return_predictions: Whether to return predictions and probabilities

    Returns:
        accuracy: Overall accuracy
        predictions: (optional) Array of predictions
        probabilities: (optional) Array of prediction probabilities
        targets: (optional) Array of true labels
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Get probabilities and predictions
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate accuracy
    accuracy = np.mean(all_predictions == all_targets)

    if return_predictions:
        return accuracy, all_predictions, all_probs, all_targets
    else:
        return accuracy


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Per-class accuracy
    if class_names is not None:
        print("\nPer-class Accuracy:")
        print("-" * 40)
        for i, class_name in enumerate(class_names):
            class_mask = y_true == i
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == i).mean()
                print(f"{class_name:15s}: {class_acc:.4f}")


# ============================================================================
# Model Analysis Utilities
# ============================================================================


def count_parameters(model):
    """
    Count total and trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    return total_params, trainable_params


def analyze_predictions(y_true, y_pred, y_probs, class_names=None, n_samples=5):
    """
    Analyze model predictions including most confident and least confident predictions

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        class_names: List of class names
        n_samples: Number of samples to show
    """
    # Get confidence scores (max probability)
    confidences = np.max(y_probs, axis=1)

    # Most confident correct predictions
    correct_mask = y_pred == y_true
    correct_confidences = confidences[correct_mask]
    correct_indices = np.where(correct_mask)[0]

    if len(correct_indices) > 0:
        most_confident_correct = correct_indices[
            np.argsort(correct_confidences)[-n_samples:]
        ]

        print(f"\nMost Confident Correct Predictions (top {n_samples}):")
        print("-" * 50)
        for idx in reversed(most_confident_correct):
            true_label = class_names[y_true[idx]] if class_names else y_true[idx]
            conf = confidences[idx]
            print(f"Sample {idx}: {true_label} (confidence: {conf:.4f})")

    # Most confident incorrect predictions
    incorrect_mask = y_pred != y_true
    incorrect_confidences = confidences[incorrect_mask]
    incorrect_indices = np.where(incorrect_mask)[0]

    if len(incorrect_indices) > 0:
        most_confident_incorrect = incorrect_indices[
            np.argsort(incorrect_confidences)[-n_samples:]
        ]

        print(f"\nMost Confident Incorrect Predictions (top {n_samples}):")
        print("-" * 50)
        for idx in reversed(most_confident_incorrect):
            true_label = class_names[y_true[idx]] if class_names else y_true[idx]
            pred_label = class_names[y_pred[idx]] if class_names else y_pred[idx]
            conf = confidences[idx]
            print(
                f"Sample {idx}: True: {true_label}, Predicted: {pred_label} "
                f"(confidence: {conf:.4f})"
            )

    # Least confident predictions
    least_confident = np.argsort(confidences)[:n_samples]

    print(f"\nLeast Confident Predictions (bottom {n_samples}):")
    print("-" * 50)
    for idx in least_confident:
        true_label = class_names[y_true[idx]] if class_names else y_true[idx]
        pred_label = class_names[y_pred[idx]] if class_names else y_pred[idx]
        conf = confidences[idx]
        correct = "✓" if y_true[idx] == y_pred[idx] else "✗"
        print(
            f"Sample {idx}: True: {true_label}, Predicted: {pred_label} "
            f"(confidence: {conf:.4f}) {correct}"
        )


# ============================================================================
# Data Processing Utilities
# ============================================================================


def create_data_splits(
    X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42, stratify=True
):
    """
    Create train/validation/test splits

    Args:
        X: Features
        y: Labels
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random state for reproducibility
        stratify: Whether to stratify splits

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, (
        "Split proportions must sum to 1"
    )

    # First split: train+val vs test
    test_split = test_size
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_split,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    # Second split: train vs val
    val_split = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_split,
        random_state=random_state,
        stratify=y_temp if stratify else None,
    )

    print(f"Data split complete:")
    print(f"  Training: {len(X_train)} samples ({train_size * 100:.0f}%)")
    print(f"  Validation: {len(X_val)} samples ({val_size * 100:.0f}%)")
    print(f"  Test: {len(X_test)} samples ({test_size * 100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_experiment_summary(results_dict, metric="accuracy"):
    """
    Print summary of experimental results

    Args:
        results_dict: Dictionary mapping experiment names to results
        metric: Metric to summarize (default: 'accuracy')
    """
    print("\nExperiment Summary")
    print("=" * 60)
    print(
        f"{'Experiment':<30} {'Best ' + metric.capitalize():<15} {'Final ' + metric.capitalize():<15}"
    )
    print("-" * 60)

    for exp_name, history in results_dict.items():
        # Find the appropriate metric key
        metric_keys = [
            f"val_{metric}",
            f"test_{metric}",
            f"val_{metric}s",
            f"test_{metric}s",
            f"{metric}s",
        ]

        metric_values = None
        for key in metric_keys:
            if key in history:
                metric_values = history[key]
                break

        if metric_values:
            best_value = max(metric_values) if "acc" in metric else min(metric_values)
            final_value = metric_values[-1]
            print(f"{exp_name:<30} {best_value:<15.4f} {final_value:<15.4f}")
        else:
            print(f"{exp_name:<30} {'N/A':<15} {'N/A':<15}")


# ============================================================================
# PyTorch Utilities
# ============================================================================


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def get_device():
    """
    Get the best available device

    Returns:
        device: torch.device object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filename: Filename to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint.pth", device="cpu"):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        filename: Filename to load
        device: Device to load to

    Returns:
        epoch: Saved epoch
        loss: Saved loss
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded from {filename} (epoch {epoch})")
    return epoch, loss


# ============================================================================
# Progress Tracking
# ============================================================================


class ProgressTracker:
    """Simple progress tracker for training loops"""

    def __init__(self, num_epochs, num_batches):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.start_time = None

    def start(self):
        """Start timing"""
        self.start_time = time.time()

    def update(self, epoch, batch, loss=None, accuracy=None):
        """Update progress"""
        import time

        elapsed = time.time() - self.start_time
        progress = (epoch * self.num_batches + batch) / (
            self.num_epochs * self.num_batches
        )
        eta = elapsed / progress - elapsed if progress > 0 else 0

        status = f"Epoch {epoch + 1}/{self.num_epochs} | Batch {batch + 1}/{self.num_batches}"
        if loss is not None:
            status += f" | Loss: {loss:.4f}"
        if accuracy is not None:
            status += f" | Acc: {accuracy:.4f}"
        status += f" | ETA: {eta / 60:.1f}m"

        print(f"\r{status}", end="")

    def finish(self):
        """Finish and print total time"""
        total_time = time.time() - self.start_time
        print(f"\nTotal training time: {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print("Available functions:")
    print("- Visualization: plot_samples, plot_training_curves, plot_confusion_matrix")
    print("- Evaluation: evaluate_model, print_classification_report")
    print("- Analysis: count_parameters, analyze_predictions")
    print("- Data: create_data_splits, print_experiment_summary")
    print("- PyTorch: set_random_seeds, get_device, save/load_checkpoint")
    print("- Progress: ProgressTracker")
