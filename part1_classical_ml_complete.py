"""
Part 1: Classical Machine Learning Solution
MNIST Classification with MLP (from scratch) and SVM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================


def load_and_preprocess_data():
    """Load MNIST data and create train/val/test splits"""
    print("Loading MNIST dataset...")
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    # Convert to appropriate types
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Create train/val/test splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================================
# Multi-Layer Perceptron Implementation
# ============================================================================


class MLPFromScratch:
    """Multi-Layer Perceptron implemented from scratch with numpy"""

    def __init__(
        self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.01
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Initialize weights using Xavier initialization
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

        # For momentum
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        # Store activations for backprop
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, X):
        """Forward propagation"""
        # X shape: (n_samples, n_features)
        # Transpose to work with column vectors
        X = X.T

        # Layer 1
        self.z1 = self.W1 @ X + self.b1
        self.a1 = self.relu(self.z1)

        # Layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2.T  # Return shape: (n_samples, n_classes)

    def compute_loss(self, y_pred, y_true):
        """Compute cross-entropy loss"""
        n_samples = y_true.shape[0]

        # Convert y_true to one-hot encoding if needed
        if y_true.ndim == 1:
            y_true_one_hot = np.zeros((n_samples, self.output_size))
            y_true_one_hot[np.arange(n_samples), y_true] = 1
            y_true = y_true_one_hot

        # Cross-entropy loss with numerical stability
        epsilon = 1e-8
        loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
        return loss

    def backward(self, X, y_true, momentum=0.9):
        """Backward propagation with momentum"""
        n_samples = X.shape[0]
        X = X.T

        # Convert y_true to one-hot
        y_true_one_hot = np.zeros((n_samples, self.output_size))
        y_true_one_hot[np.arange(n_samples), y_true] = 1
        y_true_one_hot = y_true_one_hot.T

        # Output layer gradients
        dz2 = self.a2 - y_true_one_hot
        dW2 = (1 / n_samples) * dz2 @ self.a1.T
        db2 = (1 / n_samples) * np.sum(dz2, axis=1, keepdims=True)

        # Hidden layer gradients
        da1 = self.W2.T @ dz2
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1 / n_samples) * dz1 @ X.T
        db1 = (1 / n_samples) * np.sum(dz1, axis=1, keepdims=True)

        # Update with momentum
        self.vW2 = momentum * self.vW2 - self.lr * dW2
        self.vb2 = momentum * self.vb2 - self.lr * db2
        self.vW1 = momentum * self.vW1 - self.lr * dW1
        self.vb1 = momentum * self.vb1 - self.lr * db1

        # Apply updates
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

    def train(
        self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, verbose=True
    ):
        """Train the network using mini-batch gradient descent"""
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0

            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch)

                # Backward pass
                self.backward(X_batch, y_batch)

                epoch_loss += loss

            # Validation
            val_pred = self.predict(X_val)
            val_acc = np.mean(val_pred == y_val)

            train_losses.append(epoch_loss / n_batches)
            val_accuracies.append(val_acc)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}, Loss: {epoch_loss / n_batches:.4f}, "
                    f"Val Accuracy: {val_acc:.4f}"
                )

        return train_losses, val_accuracies

    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """Return probability predictions"""
        return self.forward(X)


# ============================================================================
# Sklearn-compatible MLP Wrapper
# ============================================================================


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for our MLP implementation"""

    def __init__(self, hidden_size=128, learning_rate=0.01, epochs=50, batch_size=128):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.mlp = None

    def fit(self, X, y):
        """Fit the model"""
        # Initialize MLP
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        self.mlp = MLPFromScratch(
            input_size=n_features,
            hidden_size=self.hidden_size,
            output_size=n_classes,
            learning_rate=self.learning_rate,
        )

        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        # Train
        self.train_losses_, self.val_accuracies_ = self.mlp.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=False,
        )

        return self

    def predict(self, X):
        """Make predictions"""
        return self.mlp.predict(X)

    def predict_proba(self, X):
        """Return probability predictions"""
        return self.mlp.predict_proba(X)

    def score(self, X, y):
        """Return accuracy score"""
        return np.mean(self.predict(X) == y)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================


def train_mlp_experiments(X_train, y_train, X_val, y_val):
    """Run MLP experiments with different hyperparameters"""

    # Experiment 1: Different learning rates
    print("\n" + "=" * 60)
    print("Experiment 1: Effect of Learning Rate")
    print("=" * 60)

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    results_lr = {}

    for lr in learning_rates:
        print(f"\nTraining with learning rate = {lr}")
        mlp = MLPFromScratch(learning_rate=lr)

        # Use smaller subset for faster experimentation
        train_losses, val_accs = mlp.train(
            X_train[:5000], y_train[:5000], X_val, y_val, epochs=30, verbose=False
        )

        results_lr[lr] = {"train_losses": train_losses, "val_accuracies": val_accs}

        print(f"Final validation accuracy: {val_accs[-1]:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for lr, result in results_lr.items():
        ax1.plot(result["train_losses"], label=f"LR={lr}")
        ax2.plot(result["val_accuracies"], label=f"LR={lr}")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss vs Learning Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy vs Learning Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Experiment 2: Different batch sizes
    print("\n" + "=" * 60)
    print("Experiment 2: Effect of Batch Size")
    print("=" * 60)

    batch_sizes = [32, 128, 512]
    results_batch = {}

    for batch_size in batch_sizes:
        print(f"\nTraining with batch size = {batch_size}")
        mlp = MLPFromScratch(learning_rate=0.01)

        start_time = time.time()
        train_losses, val_accs = mlp.train(
            X_train[:5000],
            y_train[:5000],
            X_val,
            y_val,
            epochs=20,
            batch_size=batch_size,
            verbose=False,
        )
        training_time = time.time() - start_time

        results_batch[batch_size] = {
            "train_losses": train_losses,
            "val_accuracies": val_accs,
            "time": training_time,
        }

        print(f"Training time: {training_time:.2f}s")
        print(f"Final validation accuracy: {val_accs[-1]:.4f}")

    return results_lr, results_batch


def train_svm_model(X_train, y_train, X_val, y_val):
    """Train SVM with hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("Support Vector Machine Training")
    print("=" * 60)

    # Create SVM model
    svm_model = SVC(random_state=42)

    # Define parameter grid
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.001, 0.01],
        "kernel": ["rbf", "poly"],
    }

    # Use subset for grid search
    X_subset = X_train[:5000]
    y_subset = y_train[:5000]

    # Grid search with cross-validation
    print("Performing grid search...")
    grid_search = GridSearchCV(
        svm_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_subset, y_subset)
    grid_time = time.time() - start_time

    print(f"\nGrid search completed in {grid_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Train final model on full dataset
    print("\nTraining final SVM model on full dataset...")
    best_svm = SVC(**grid_search.best_params_, random_state=42)

    start_time = time.time()
    best_svm.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    val_acc = best_svm.score(X_val, y_val)
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Validation accuracy: {val_acc:.4f}")

    return best_svm, grid_search


def create_sklearn_pipeline():
    """Create complete sklearn pipeline with preprocessing"""
    print("\n" + "=" * 60)
    print("Creating Sklearn Pipeline")
    print("=" * 60)

    # Create pipeline
    pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("classifier", SVC(kernel="rbf", C=10, gamma=0.001, random_state=42)),
        ]
    )

    return pipeline


def compare_models(X_test, y_test, mlp_model, svm_model):
    """Compare MLP and SVM performance"""
    print("\n" + "=" * 60)
    print("Model Comparison on Test Set")
    print("=" * 60)

    # MLP predictions
    mlp_pred = mlp_model.predict(X_test)
    mlp_acc = np.mean(mlp_pred == y_test)

    # SVM predictions
    svm_pred = svm_model.predict(X_test)
    svm_acc = np.mean(svm_pred == y_test)

    print(f"MLP Test Accuracy: {mlp_acc:.4f}")
    print(f"SVM Test Accuracy: {svm_acc:.4f}")

    # Detailed classification reports
    print("\nMLP Classification Report:")
    print(classification_report(y_test, mlp_pred))

    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_pred))

    # Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # MLP confusion matrix
    cm_mlp = confusion_matrix(y_test, mlp_pred)
    im1 = ax1.imshow(cm_mlp, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("MLP Confusion Matrix")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    plt.colorbar(im1, ax=ax1)

    # Add text annotations
    for i in range(10):
        for j in range(10):
            ax1.text(j, i, str(cm_mlp[i, j]), ha="center", va="center")

    # SVM confusion matrix
    cm_svm = confusion_matrix(y_test, svm_pred)
    im2 = ax2.imshow(cm_svm, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.set_title("SVM Confusion Matrix")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    plt.colorbar(im2, ax=ax2)

    # Add text annotations
    for i in range(10):
        for j in range(10):
            ax2.text(j, i, str(cm_svm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()

    return mlp_acc, svm_acc


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main execution function"""

    # Load and preprocess data
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    ) = load_and_preprocess_data()

    # Visualize some samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    for i in range(10):
        axes[i].imshow(X_train[i].reshape(28, 28), cmap="gray")
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis("off")
    plt.suptitle("Sample MNIST Digits")
    plt.tight_layout()
    plt.show()

    # Run MLP experiments
    results_lr, results_batch = train_mlp_experiments(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    # Train final MLP model
    print("\n" + "=" * 60)
    print("Training Final MLP Model")
    print("=" * 60)

    final_mlp = MLPClassifier(
        hidden_size=128, learning_rate=0.01, epochs=30, batch_size=128
    )
    final_mlp.fit(
        X_train_scaled[:20000], y_train[:20000]
    )  # Use subset for reasonable time

    # Train SVM model
    svm_model, grid_search = train_svm_model(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    # Create and demonstrate pipeline
    pipeline = create_sklearn_pipeline()
    print("\nTraining pipeline...")
    pipeline.fit(X_train[:20000], y_train[:20000])  # Use subset
    pipeline_acc = pipeline.score(X_val, y_val)
    print(f"Pipeline validation accuracy: {pipeline_acc:.4f}")

    # Compare models on test set
    compare_models(X_test_scaled, y_test, final_mlp, svm_model)

    # Cross-validation example
    print("\n" + "=" * 60)
    print("Cross-Validation Example")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        final_mlp,
        X_train_scaled[:5000],
        y_train[:5000],
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return final_mlp, svm_model, pipeline


if __name__ == "__main__":
    # Run the complete solution
    mlp_model, svm_model, pipeline = main()

    print("\n" + "=" * 60)
    print("Part 1 Complete!")
    print("=" * 60)
