"""Perceptron for classification with augmented data.

This script implements a simple perceptron for spectral data classification,
focusing on classification of num_classe using augmented data.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time  # Import for measuring training time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import spmatrix

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================
# Number of training epochs
N_EPOCHS = 250
# Random seed for reproducibility
RANDOM_SEED = 42
# Batch size for training
BATCH_SIZE = 32
# Learning rate for the optimizer
LEARNING_RATE = 0.0001
# Proportion of data for the test set
TEST_SIZE = 0.25
# Proportion of data for the validation set (relative to remaining data)
VAL_SIZE = 0.2
# ==============================================================================

# Get current notebook path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataLoading import data_3cl, spectral_cols
from src.data_augmenter import DataAugmenter, AugmentationParams


@dataclass
class ClassificationMetrics:
    """Class to store training metrics for classification at each epoch"""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    epoch_time: float = 0.0  # Epoch execution time in seconds


def plot_classification_metrics(
    metrics_list: List[ClassificationMetrics],
    title: str = "Metrics Evolution",
    save_path: Optional[str] = None,
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


# Simple Perceptron model for multiclass classification
class MultiClassPerceptron(nn.Module):
    """Single layer perceptron for multiclass classification

    Args:
        input_dim: Number of input features
        num_classes: Number of classes to predict
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No need to add softmax here as CrossEntropyLoss includes it
        return self.linear(x)


# =============================================================================
# Perceptron for classification with data augmentation
# =============================================================================
print("\n" + "=" * 80)
print("PERCEPTRON FOR CLASSIFICATION (NUM_CLASSE PREDICTION) WITH DATA AUGMENTATION")
print("=" * 80)

# Encoding classes with LabelEncoder for consistency
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(data_3cl["num_classe"])
all_labels_encoded = np.array(all_labels_encoded)  # Convert to numpy array explicitly

# Extract encoding mapping for future reference
classes = label_encoder.classes_
class_mapping = {i: cls for i, cls in enumerate(classes)}
num_classes = len(classes)

print(f"Number of classes: {num_classes}")
print(f"Classes: {classes.tolist()}")
print("Class index mapping:")
for idx, class_name in class_mapping.items():
    print(f"  {idx} -> {class_name}")

# Preparing data for classification
indices = np.arange(len(data_3cl))
indices_train_temp, indices_test, y_train_temp, y_test_temp = train_test_split(
    indices,
    all_labels_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=all_labels_encoded,  # Ensure balanced class distribution
)

indices_train, indices_val = train_test_split(
    indices_train_temp,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_train_temp,  # Maintain class distribution
)

# Initial class distribution
print("\nInitial class distribution:")
for i, cls in enumerate(classes):
    count = np.sum(all_labels_encoded == i)
    print(f"Class {i} ({cls}): {count} samples")

# Display set sizes
print(f"\nTraining set (original): {len(indices_train)} samples")
print(f"Validation set: {len(indices_val)} samples")
print(f"Test set: {len(indices_test)} samples")

# Create datasets before augmentation
train_data = data_3cl.iloc[indices_train].copy()
val_data = data_3cl.iloc[indices_val].copy()
test_data = data_3cl.iloc[indices_test].copy()

# Create augmentation parameters
augmentation_params = AugmentationParams(
    mixup_alpha=0.4,  # Higher alpha for more diverse mixing
    gaussian_noise_std=0.03,  # Increased noise for better robustness
    jitter_factor=0.04,  # More intensity variation
    augmentation_probability=0.8,  # Higher probability of applying augmentation
    by=["symptom", "num_classe"],  # Group by these columns
    batch_size=100,  # Generate 100 samples per group
    exclude_columns=None,  # Don't exclude any columns to keep all spectral data
)

# Create augmenter and augment training data
augmenter = DataAugmenter(augmentation_params)
train_data_augmented = augmenter.augment(train_data)

print(f"Training set (after augmentation): {len(train_data_augmented)} samples")

# Prepare features and targets for different sets
X_train = np.array(train_data_augmented[spectral_cols])
y_train = np.array(label_encoder.transform(train_data_augmented["num_classe"]))

X_val = np.array(val_data[spectral_cols])
y_val = np.array(label_encoder.transform(val_data["num_classe"]))

X_test = np.array(test_data[spectral_cols])
y_test = np.array(label_encoder.transform(test_data["num_classe"]))

# Standardize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Create datasets and loaders
train_dataset = SpectralDataset(X_train_scaled, y_train.astype(np.float32))
val_dataset = SpectralDataset(X_val_scaled, y_val.astype(np.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function and optimizer
model = MultiClassPerceptron(input_dim=len(spectral_cols), num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics
metrics_history: List[ClassificationMetrics] = []

print("Starting perceptron training for classification with augmentation...")
total_training_start = time.time()
for epoch in range(N_EPOCHS):
    epoch_start = time.time()
    # Set model to training mode
    model.train()
    # Initialize the total training loss for this epoch
    train_loss = 0.0
    # Iterate over mini-batches of training data
    for X_batch, y_batch in train_loader:
        # Convert labels to long for CrossEntropyLoss
        y_batch = y_batch.long()

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

    # Validation phase: evaluate model performance on validation set
    model.eval()
    # Initialize metrics
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        # Iterate over mini-batches of validation data
        for X_batch, y_batch in val_loader:
            # Convert labels to long for CrossEntropyLoss
            y_batch = y_batch.long()

            # Forward pass only
            outputs = model(X_batch)
            # Accumulate the validation loss
            val_loss += criterion(outputs, y_batch).item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    # Calculate average metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    # Calculate epoch execution time
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    # Store metrics for each epoch
    metrics_history.append(
        ClassificationMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            epoch_time=epoch_time,
        )
    )

    # Keep track of the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()

    # Print metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {total_training_time/N_EPOCHS:.2f} seconds")

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    best_epoch = [m.epoch for m in metrics_history if m.val_loss == best_val_loss][0]
    print(
        f"Best model loaded from epoch {best_epoch} with validation loss of {best_val_loss:.4f}"
    )

# Evaluation on test set
model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    # Convert test data to tensors directly
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    # Get predictions for all test data at once
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    test_total = y_test_tensor.size(0)
    test_correct = (predicted == y_test_tensor).sum().item()

    # Store predictions and targets for detailed metrics
    all_predictions = predicted.cpu().numpy()
    all_targets = y_test_tensor.cpu().numpy()

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

print("\nClassification Report:")
cls_report = classification_report(
    all_targets, all_predictions, target_names=class_names, digits=4
)
print(cls_report)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
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
plt.savefig(
    str(project_root) + "/plots/confusion_matrix_perceptron_classification.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Visualize the evolution of metrics
plot_classification_metrics(
    metrics_history,
    title="Evolution of Training Metrics for Classification",
    save_path=str(project_root)
    + "/plots/training_metrics_perceptron_classification.png",
)

# Plot training loss vs. validation accuracy
plt.figure(figsize=(10, 6))
epochs = [m.epoch for m in metrics_history]
train_losses = [m.train_loss for m in metrics_history]
val_accuracies = [m.val_accuracy for m in metrics_history]

# Create two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Plot training loss on the first y-axis
ax1.plot(epochs, train_losses, "b-", label="Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="b")
ax1.tick_params(axis="y", labelcolor="b")

# Plot validation accuracy on the second y-axis
ax2.plot(epochs, val_accuracies, "r-", label="Validation Accuracy")
ax2.set_ylabel("Accuracy (%)", color="r")
ax2.tick_params(axis="y", labelcolor="r")

# Add grid and legend
ax1.grid(True, linestyle="--", alpha=0.7)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Training Loss vs. Validation Accuracy")
plt.tight_layout()
plt.savefig(
    str(project_root) + "/plots/loss_vs_accuracy_perceptron_classification.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
