"""Perceptron for regression.

This script implements a simple perceptron for spectral data regression,
focusing on Chl prediction.
"""

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    r2_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast, Dict, Any
import os
import time  # Import for measuring training time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import spmatrix
import numpy as np

# Get current notebook path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataLoading import data_3cl, spectral_cols


# Create directory for plots
plots_dir = os.path.join(project_root, "plots")
os.makedirs(plots_dir, exist_ok=True)


@dataclass
class RegressionMetrics:
    """Class to store training metrics for regression at each epoch"""

    epoch: int
    train_loss: float
    val_loss: float
    epoch_time: float = 0.0  # Epoch execution time in seconds


@dataclass
class ClassificationMetrics:
    """Class to store training metrics for classification at each epoch"""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    epoch_time: float = 0.0  # Epoch execution time in seconds


def plot_regression_metrics(
    metrics_list: List[RegressionMetrics],
    title: str = "Metrics Evolution",
    save_path: Optional[str] = None,
) -> None:
    """
    Displays the evolution of training metrics across epochs for regression models

    Args:
        metrics_list: List of RegressionMetrics objects
        title: Chart title
        save_path: Path to save the chart (None = no saving)
    """
    epochs = [m.epoch for m in metrics_list]
    train_losses = [m.train_loss for m in metrics_list]
    val_losses = [m.val_loss for m in metrics_list]
    epoch_times = [m.epoch_time for m in metrics_list]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [2, 1]}
    )

    # First subplot: training and validation losses
    ax1.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="red", marker="x")
    ax1.set_title(title)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Second subplot: training time per epoch
    ax2.bar(epochs, epoch_times, color="green", alpha=0.7)
    ax2.set_title("Training Time per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Total and average training time
    total_time = sum(epoch_times)
    avg_time = total_time / len(epoch_times) if epoch_times else 0
    ax2.text(
        0.02,
        0.95,
        f"Total time: {total_time:.2f}s\nAverage time: {avg_time:.2f}s/epoch",
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Appearance improvement
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_classification_metrics(
    metrics_list: List[ClassificationMetrics],
    title: str = "Metrics Evolution",
    save_path: str | None = None,
) -> None:
    """
    Displays the evolution of training metrics across epochs for classification models

    Args:
        metrics_list: List of ClassificationMetrics objects
        title: Chart title
        save_path: Path to save the chart (None = no saving)
    """
    epochs = [m.epoch for m in metrics_list]
    train_losses = [m.train_loss for m in metrics_list]
    val_losses = [m.val_loss for m in metrics_list]
    val_accuracies = [m.val_accuracy for m in metrics_list]
    epoch_times = [m.epoch_time for m in metrics_list]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 18), gridspec_kw={"height_ratios": [2, 2, 1]}
    )

    # First subplot: training and validation losses
    ax1.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="red", marker="x")
    ax1.set_title(title)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Second subplot: validation accuracy
    ax2.plot(
        epochs, val_accuracies, label="Validation Accuracy", color="purple", marker="d"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Third subplot: training time per epoch
    ax3.bar(epochs, epoch_times, color="green", alpha=0.7)
    ax3.set_title("Training Time per Epoch")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time (seconds)")
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Total and average training time
    total_time = sum(epoch_times)
    avg_time = total_time / len(epoch_times) if epoch_times else 0
    ax3.text(
        0.02,
        0.95,
        f"Total time: {total_time:.2f}s\nAverage time: {avg_time:.2f}s/epoch",
        transform=ax3.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Appearance improvement
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """
    Display a confusion matrix with improved formatting

    Args:
        cm: Confusion matrix
        class_names: Class names
        title: Chart title
        save_path: Path to save the chart (None = no saving)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    # Appearance improvement
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual Values",
    save_path: str | None = None,
) -> None:
    """
    Display a scatter plot of predictions vs actual values for a regression model

    Args:
        y_true: Actual values
        y_pred: Model predictions
        title: Chart title
        save_path: Path to save the chart (None = no saving)
    """
    plt.figure(figsize=(10, 8))

    # Calculate statistics
    r2 = r2_score(y_true, y_pred)

    # Plot points
    plt.scatter(y_true, y_pred, alpha=0.6)

    # Plot ideal line (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.title(f"{title}\nR² = {r2:.4f}")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Appearance improvement
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# Custom Dataset class for our spectral data
class SpectralDataset(Dataset):
    """Simple Dataset class to handle spectral data for PyTorch"""

    def __init__(self, X: np.ndarray | spmatrix, y: np.ndarray | spmatrix):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# Simple Perceptron model (one d
# ense layer)
class Perceptron(nn.Module):
    """Single layer perceptron for regression

    Args:
        input_dim: Number of input features
        output_dim: Number of outputs (1 for single value regression)
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# =============================================================================
# Part 1: Perceptron for regression
# =============================================================================
print("\n" + "=" * 80)
print("PERCEPTRON FOR REGRESSION (CHL PREDICTION)")
print("=" * 80)

# Preparing data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    data_3cl[spectral_cols],
    data_3cl[["Chl", "num_classe"]],
    test_size=0.25,
    random_state=42,
)

# Split remaining data into train and validation sets
X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
    X_train_reg, y_train_reg, test_size=0.2, random_state=42
)

# Standardize features for regression
scaler_X_reg = StandardScaler()
scaler_y_Chl = StandardScaler()

X_train_reg = scaler_X_reg.fit_transform(X_train_reg)
X_val_reg = scaler_X_reg.transform(X_val_reg)
X_test_reg = scaler_X_reg.transform(X_test_reg)

y_train_chl = scaler_y_Chl.fit_transform(y_train_reg["Chl"].values.reshape(-1, 1))
y_val_chl = scaler_y_Chl.transform(y_val_reg["Chl"].values.reshape(-1, 1))
y_test_chl = scaler_y_Chl.transform(y_test_reg["Chl"].values.reshape(-1, 1))

# Create datasets and loaders for training and validation only
train_dataset = SpectralDataset(X_train_reg, y_train_chl)
val_dataset = SpectralDataset(X_val_reg, y_val_chl)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model, loss function and optimizer
model = Perceptron(input_dim=len(spectral_cols))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 150
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics
metrics_history: List[RegressionMetrics] = []

print("Starting perceptron training...")
total_training_start = time.time()
for epoch in range(n_epochs):
    epoch_start = time.time()
    # Set model to training mode
    model.train()
    # Initialize the total training loss for this epoch
    train_loss = 0.0
    # Iterate over mini-batches of training data
    for X_batch, y_batch in train_loader:
        # Reset gradients to zero before computing new gradients
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        outputs = model(X_batch)
        # Compute the loss between predictions and true values
        loss = criterion(outputs, y_batch)
        # Backward pass: compute gradients of the loss
        loss.backward()
        # Update model parameters using the optimizer
        optimizer.step()
        # Accumulate the batch loss
        train_loss += loss.item()

    # Validation phase
    model.eval()
    # Initialize the total validation loss for this epoch
    val_loss = 0.0
    # Disable gradient computation for validation
    with torch.no_grad():
        # Iterate over mini-batches of validation data
        for X_batch, y_batch in val_loader:
            # Forward pass only
            outputs = model(X_batch)
            # Accumulate the validation loss
            val_loss += criterion(outputs, y_batch).item()

    # Calculate average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    # Calculate epoch execution time
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    # Record metrics for this epoch
    metrics_history.append(
        RegressionMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            epoch_time=epoch_time,
        )
    )

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()

    # Print metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {total_training_time/n_epochs:.2f} seconds")

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Display metrics evolution
plot_regression_metrics(
    metrics_history,
    title="Metrics Evolution for Perceptron (Regression)",
    save_path=os.path.join(plots_dir, "perceptron_regression_metrics.png"),
)

# Evaluation on test set - simplified version without DataLoader
model.eval()
with torch.no_grad():
    # Convert test data to tensors directly
    X_test_tensor = torch.FloatTensor(X_test_reg)
    y_test_tensor = torch.FloatTensor(y_test_chl)

    # Get predictions for all test data at once
    test_predictions = model(X_test_tensor).numpy()
    test_targets = y_test_tensor.numpy()

# Convert predictions back to original scale
test_predictions = scaler_y_Chl.inverse_transform(test_predictions)
test_targets = scaler_y_Chl.inverse_transform(test_targets)

# Calculate R² score
r2 = r2_score(test_targets, test_predictions)
print(f"\nR² score on test set: {r2:.4f}")

# Display predictions vs actual values
plot_regression_predictions(
    cast(np.ndarray, test_targets),
    cast(np.ndarray, test_predictions),
    title="Predictions vs Actual Values - Perceptron (Regression)",
    save_path=os.path.join(plots_dir, "perceptron_regression_predictions.png"),
)

# =============================================================================
# Part 2: Multiclass Perceptron with Softmax + CrossEntropy and LabelEncoder
# =============================================================================
print("\n" + "=" * 80)
print("MULTICLASS PERCEPTRON WITH LABELENCODER")
print("=" * 80)


# Class for the multiclass perceptron model
class MultiClassPerceptron(nn.Module):
    """Perceptron for multiclass classification

    Args:
        input_dim: Number of input features
        num_classes: Number of classes to predict
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No need to add softmax here as CrossEntropyLoss already includes it
        return self.linear(x)


# Encode classes with LabelEncoder for consistency
label_encoder = LabelEncoder()
# Encoding on the entire dataset before splitting
all_labels_encoded = label_encoder.fit_transform(data_3cl["num_classe"])

# Extract encoding mapping for future reference
classes = label_encoder.classes_
class_mapping = {i: cls for i, cls in enumerate(classes)}
print(f"Number of classes: {len(classes)}")
print(f"Classes: {classes.tolist()}")
print("Class index mapping:")
for idx, class_name in class_mapping.items():
    print(f"  {idx} -> {class_name}")

# Preparing data for classification
X_cls = data_3cl[spectral_cols]
y_cls = all_labels_encoded

# Split into training, validation, and test sets
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42
)

X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
    X_train_cls, y_train_cls, test_size=0.2, random_state=42
)

# Standardize features
scaler_X_cls = StandardScaler()
X_train_scaled = scaler_X_cls.fit_transform(X_train_cls)
X_val_scaled = scaler_X_cls.transform(X_val_cls)
X_test_scaled = scaler_X_cls.transform(X_test_cls)

# Create datasets and dataloaders
# Note: Convert to float32 for compatibility with SpectralDataset
train_dataset_cls = SpectralDataset(X_train_scaled, y_train_cls.astype(np.float32))
val_dataset_cls = SpectralDataset(X_val_scaled, y_val_cls.astype(np.float32))
test_dataset_cls = SpectralDataset(X_test_scaled, y_test_cls.astype(np.float32))

train_loader_cls = DataLoader(train_dataset_cls, batch_size=32, shuffle=True)
val_loader_cls = DataLoader(val_dataset_cls, batch_size=32)
test_loader_cls = DataLoader(test_dataset_cls, batch_size=32)

# Initialize model, loss function, and optimizer
num_classes = len(classes)
model_cls = MultiClassPerceptron(input_dim=len(spectral_cols), num_classes=num_classes)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.001)

# Training loop
n_epochs = 100
best_val_loss = float("inf")
best_model = None

# List to store training metrics
metrics_history_cls: List[ClassificationMetrics] = []

print("\nStarting multiclass perceptron training...")
total_training_start = time.time()
for epoch in range(n_epochs):
    epoch_start = time.time()
    # Training mode
    model_cls.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader_cls:
        # Convert labels for CrossEntropyLoss
        y_batch = y_batch.long()

        optimizer_cls.zero_grad()
        outputs = model_cls(X_batch)
        loss = criterion_cls(outputs, y_batch)
        loss.backward()
        optimizer_cls.step()
        train_loss += loss.item()

    # Validation mode
    model_cls.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader_cls:
            # Convert labels for CrossEntropyLoss
            y_batch = y_batch.long()

            outputs = model_cls(X_batch)
            loss = criterion_cls(outputs, y_batch)
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    # Calculate average metrics
    avg_train_loss = train_loss / len(train_loader_cls)
    avg_val_loss = val_loss / len(val_loader_cls)
    val_accuracy = val_correct / val_total

    # Calculate epoch execution time
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    # Record metrics
    metrics_history_cls.append(
        ClassificationMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            epoch_time=epoch_time,
        )
    )

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = model_cls.state_dict().copy()

    # Print metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {total_training_time/n_epochs:.2f} seconds")

# Display metrics evolution
plot_classification_metrics(
    metrics_history_cls,
    title="Metrics Evolution for Perceptron (Classification)",
    save_path=os.path.join(plots_dir, "perceptron_classification_metrics.png"),
)

# Load the best model for evaluation
if best_model is not None:
    model_cls.load_state_dict(best_model)
else:
    print("Warning: No best model was saved during training.")

# Evaluation on test set
model_cls.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader_cls:
        # Convert labels for CrossEntropyLoss
        y_batch = y_batch.long()

        outputs = model_cls(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        test_total += y_batch.size(0)
        test_correct += (predicted == y_batch).sum().item()

        # Save predictions and targets for detailed metrics
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

# Calculate and display performance metrics
test_accuracy = 100 * test_correct / test_total
balanced_acc = balanced_accuracy_score(all_targets, all_predictions)
conf_matrix = confusion_matrix(all_targets, all_predictions)

print("\n" + "=" * 50)
print("CLASSIFICATION RESULTS")
print("=" * 50)
print(f"Accuracy: {test_accuracy:.2f}%")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# Use class names from LabelEncoder
class_names = [str(cls) for cls in label_encoder.classes_.tolist()]
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
plot_confusion_matrix(
    conf_matrix,
    class_names=class_names,
    title="Confusion Matrix - Perceptron (Classification)",
    save_path=os.path.join(plots_dir, "perceptron_classification_confusion.png"),
)

print("\nClassification Report:")
cls_report = classification_report(
    all_targets, all_predictions, target_names=class_names, digits=4
)
print(cls_report)
