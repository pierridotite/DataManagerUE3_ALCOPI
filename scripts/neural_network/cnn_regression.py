"""Convolutional Neural Network for regression of Chl.

This script implements a CNN for spectral data regression, with both
standard and data-augmented versions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time  # Import for measuring training time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import spmatrix

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================
# Number of training epochs
N_EPOCHS = 150
# Random seed for reproducibility
RANDOM_SEED = 42
# Batch size for training
BATCH_SIZE = 32
# Learning rate for the optimizer
LEARNING_RATE = 0.00001
# Proportion of data for the test set
TEST_SIZE = 0.25
# Proportion of data for the validation set (relative to remaining data)
VAL_SIZE = 0.2
# Kernel size for convolutional layers
KERNEL_SIZE = 3
# Stride for convolutional layers
STRIDE = 1
# Padding for convolutional layers
PADDING = 2
# ==============================================================================

# Get current notebook path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataLoading import data_3cl, spectral_cols
from src.data_augmenter import DataAugmenter, AugmentationParams


@dataclass
class TrainingMetrics:
    """Class to store training metrics at each epoch"""

    epoch: int
    train_loss: float
    val_loss: float
    epoch_time: float = 0.0  # Epoch execution time in seconds


def plot_training_metrics(
    metrics_list: List[TrainingMetrics],
    title: str = "Metrics Evolution",
    save_path: Optional[str] = None,
) -> None:
    """
    Displays the evolution of training metrics across epochs

    Args:
        metrics_list: List of TrainingMetrics objects containing metrics for each epoch
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


def print_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, set_name: str = "test"
) -> None:
    """
    Print standard regression metrics

    Args:
        y_true: Actual values
        y_pred: Model predictions
        set_name: Name of the dataset for reporting (e.g., 'train', 'test')
    """
    # Calculate R² score
    r2 = r2_score(y_true, y_pred)
    print(f"\nR² score on {set_name} set: {r2:.4f}")

    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE on {set_name} set: {mse:.4f}")

    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"RMSE on {set_name} set: {rmse:.4f}")

    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE on {set_name} set: {mae:.4f}")


# Custom Dataset class for our spectral data
class SpectralDataset(Dataset):
    """Dataset class to handle spectral data for PyTorch"""

    def __init__(self, X: np.ndarray | spmatrix, y: np.ndarray | spmatrix):
        # Reshape X for CNN input: [batch_size, channels, sequence_length]
        # For spectral data, we use 1 channel
        self.X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# CNN model for regression
class CNNModel(nn.Module):
    """CNN model with two convolutional layers for spectral data regression

    Args:
        input_dim: Length of the spectral sequence
        output_dim: Number of outputs (1 for single value regression)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        kernel_size: int = KERNEL_SIZE,
        stride: int = STRIDE,
        padding: int = PADDING,
    ):
        super().__init__()

        # Calculate output size of first conv layer
        conv1_output_size = ((input_dim + 2 * padding - kernel_size) // stride) + 1

        # Calculate output size after pooling
        pool1_output_size = conv1_output_size // 2

        # Calculate output size of second conv layer
        conv2_output_size = (
            (pool1_output_size + 2 * padding - kernel_size) // stride
        ) + 1

        # Calculate output size after pooling
        pool2_output_size = conv2_output_size // 2

        # First convolutional layer: in_channels=1 (grayscale), out_channels=16
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Second convolutional layer: in_channels=16, out_channels=32
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Final output is a flattened tensor of size (batch_size, 32 * pool2_output_size)
        # Fully connected layers for regression
        self.fc = nn.Sequential(
            nn.Linear(32 * pool2_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, 1, sequence_length)
        x = self.conv1(x)
        # Shape: (batch_size, 16, sequence_length/2)
        x = self.conv2(x)
        # Shape: (batch_size, 32, sequence_length/4)

        # Flatten: (batch_size, 32 * sequence_length/4)
        x = x.view(x.size(0), -1)

        # Output: (batch_size, output_dim)
        return self.fc(x)


# =============================================================================
# CNN for regression without data augmentation
# =============================================================================
print("\n" + "=" * 80)
print("CNN FOR REGRESSION (CHL PREDICTION) WITHOUT DATA AUGMENTATION")
print("=" * 80)

# Preparing data for regression
indices = np.arange(len(data_3cl))
indices_train_temp, indices_test = train_test_split(
    indices, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

indices_train, indices_val = train_test_split(
    indices_train_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED
)

# Display set sizes
print(f"Training set: {len(indices_train)} samples")
print(f"Validation set: {len(indices_val)} samples")
print(f"Test set: {len(indices_test)} samples")

# Create datasets
train_data = data_3cl.iloc[indices_train].copy()
val_data = data_3cl.iloc[indices_val].copy()
test_data = data_3cl.iloc[indices_test].copy()

# Prepare features and targets for different sets
X_train_reg = np.array(train_data[spectral_cols])
y_train_reg = np.array(train_data["Chl"]).reshape(-1, 1)

X_val_reg = np.array(val_data[spectral_cols])
y_val_reg = np.array(val_data["Chl"]).reshape(-1, 1)

X_test_reg = np.array(test_data[spectral_cols])
y_test_reg = np.array(test_data["Chl"]).reshape(-1, 1)

# Standardize features for regression
scaler_X_reg = StandardScaler()
scaler_y_Chl = StandardScaler()

X_train_reg = scaler_X_reg.fit_transform(X_train_reg)
X_val_reg = scaler_X_reg.transform(X_val_reg)
X_test_reg = scaler_X_reg.transform(X_test_reg)

y_train_chl = scaler_y_Chl.fit_transform(y_train_reg)
y_val_chl = scaler_y_Chl.transform(y_val_reg)
y_test_chl = scaler_y_Chl.transform(y_test_reg)

# Create datasets and loaders for training and validation only
train_dataset = SpectralDataset(X_train_reg, y_train_chl)
val_dataset = SpectralDataset(X_val_reg, y_val_chl)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function and optimizer
model = CNNModel(input_dim=len(spectral_cols))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics
metrics_history: List[TrainingMetrics] = []

print("Starting CNN training for regression without augmentation...")
total_training_start = time.time()
for epoch in range(N_EPOCHS):
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

    # Store metrics for each epoch
    metrics_history.append(
        TrainingMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
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
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {total_training_time/N_EPOCHS:.2f} seconds")

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Best model loaded with validation loss of {best_val_loss:.4f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    # Convert test data to tensors directly
    X_test_tensor = torch.FloatTensor(X_test_reg).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.FloatTensor(y_test_chl)

    # Get predictions for all test data at once
    test_predictions = model(X_test_tensor).detach().numpy()
    test_targets = y_test_tensor.detach().numpy()

# Convert predictions back to original scale
test_predictions = scaler_y_Chl.inverse_transform(test_predictions)
test_targets = scaler_y_Chl.inverse_transform(test_targets)

# Print evaluation metrics
print_regression_metrics(test_targets, test_predictions, "test")

# Visualize the evolution of metrics
plot_training_metrics(
    metrics_history,
    title="Evolution of Training Metrics for CNN Regression (Without Augmentation)",
    save_path=str(project_root) + "/plots/training_metrics_cnn_regression.png",
)

# Visualize predictions vs actual values
plt.figure(figsize=(10, 8))
plt.scatter(test_targets, test_predictions, alpha=0.6)
plt.plot(
    [np.min(test_targets), np.max(test_targets)],
    [np.min(test_targets), np.max(test_targets)],
    "r--",
)
plt.xlabel("Actual Chl Values")
plt.ylabel("Predicted Chl Values")
plt.title(
    f"CNN Regression: Predictions vs Actual Values\nR² = {r2_score(test_targets, test_predictions):.4f}"
)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(
    str(project_root) + "/plots/cnn_regression_predictions.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# =============================================================================
# CNN for regression with data augmentation
# =============================================================================
print("\n" + "=" * 80)
print("CNN FOR REGRESSION (CHL PREDICTION) WITH DATA AUGMENTATION")
print("=" * 80)

# Re-use the same train-test-validation split
train_data = data_3cl.iloc[indices_train].copy()
val_data = data_3cl.iloc[indices_val].copy()
test_data = data_3cl.iloc[indices_test].copy()

# Create augmentation parameters
augmentation_params = AugmentationParams(
    mixup_alpha=0.4,  # Higher alpha for more diverse mixing
    gaussian_noise_std=0.03,  # Increased noise for better robustness
    jitter_factor=0.04,  # More intensity variation
    augmentation_probability=0.8,  # Higher probability of applying augmentation
    by=["symptom", "variety", "plotLocation"],  # Group by these columns
    batch_size=100,  # Generate 100 samples per group
    exclude_columns=None,  # Don't exclude any columns to keep all spectral data
)

# Create augmenter and augment training data
augmenter = DataAugmenter(augmentation_params)
train_data_augmented = augmenter.augment(train_data)

print(f"Training set (original): {len(train_data)} samples")
print(f"Training set (after augmentation): {len(train_data_augmented)} samples")
print(f"Validation set: {len(val_data)} samples")
print(f"Test set: {len(test_data)} samples")

# Prepare features and targets for different sets with augmented data
X_train_reg = np.array(train_data_augmented[spectral_cols])
y_train_reg = np.array(train_data_augmented["Chl"]).reshape(-1, 1)

X_val_reg = np.array(val_data[spectral_cols])
y_val_reg = np.array(val_data["Chl"]).reshape(-1, 1)

X_test_reg = np.array(test_data[spectral_cols])
y_test_reg = np.array(test_data["Chl"]).reshape(-1, 1)

# Standardize features for regression
scaler_X_reg = StandardScaler()
scaler_y_Chl = StandardScaler()

X_train_reg = scaler_X_reg.fit_transform(X_train_reg)
X_val_reg = scaler_X_reg.transform(X_val_reg)
X_test_reg = scaler_X_reg.transform(X_test_reg)

y_train_chl = scaler_y_Chl.fit_transform(y_train_reg)
y_val_chl = scaler_y_Chl.transform(y_val_reg)
y_test_chl = scaler_y_Chl.transform(y_test_reg)

# Create datasets and loaders for training and validation only
train_dataset = SpectralDataset(X_train_reg, y_train_chl)
val_dataset = SpectralDataset(X_val_reg, y_val_chl)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function and optimizer
model_aug = CNNModel(input_dim=len(spectral_cols))
criterion = nn.MSELoss()
optimizer = optim.Adam(model_aug.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics
metrics_history: List[TrainingMetrics] = []

print("Starting CNN training for regression with augmentation...")
total_training_start = time.time()
for epoch in range(N_EPOCHS):
    epoch_start = time.time()
    # Set model to training mode
    model_aug.train()
    # Initialize the total training loss for this epoch
    train_loss = 0.0
    # Iterate over mini-batches of training data
    for X_batch, y_batch in train_loader:
        # Reset gradients to zero before computing new gradients
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        outputs = model_aug(X_batch)
        # Compute the loss between predictions and true values
        loss = criterion(outputs, y_batch)
        # Backward pass: compute gradients of the loss
        loss.backward()
        # Update model parameters using the optimizer
        optimizer.step()
        # Accumulate the batch loss
        train_loss += loss.item()

    # Validation phase
    model_aug.eval()
    # Initialize the total validation loss for this epoch
    val_loss = 0.0
    # Disable gradient computation for validation
    with torch.no_grad():
        # Iterate over mini-batches of validation data
        for X_batch, y_batch in val_loader:
            # Forward pass only
            outputs = model_aug(X_batch)
            # Accumulate the validation loss
            val_loss += criterion(outputs, y_batch).item()

    # Calculate average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    # Calculate epoch execution time
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    # Store metrics for each epoch
    metrics_history.append(
        TrainingMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            epoch_time=epoch_time,
        )
    )

    # Keep track of the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model_aug.state_dict().copy()

    # Print metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {total_training_time/N_EPOCHS:.2f} seconds")

# Load the best model
if best_model_state is not None:
    model_aug.load_state_dict(best_model_state)
    print(f"Best model loaded with validation loss of {best_val_loss:.4f}")

# Evaluation on test set
model_aug.eval()
with torch.no_grad():
    # Convert test data to tensors directly
    X_test_tensor = torch.FloatTensor(X_test_reg).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.FloatTensor(y_test_chl)

    # Get predictions for all test data at once
    test_predictions = model_aug(X_test_tensor).detach().numpy()
    test_targets = y_test_tensor.detach().numpy()

# Convert predictions back to original scale
test_predictions = scaler_y_Chl.inverse_transform(test_predictions)
test_targets = scaler_y_Chl.inverse_transform(test_targets)

# Print evaluation metrics
print_regression_metrics(test_targets, test_predictions, "test")

# Visualize the evolution of metrics
plot_training_metrics(
    metrics_history,
    title="Evolution of Training Metrics for CNN Regression (With Augmentation)",
    save_path=str(project_root)
    + "/plots/training_metrics_cnn_regression_augmented.png",
)

# Visualize predictions vs actual values
plt.figure(figsize=(10, 8))
plt.scatter(test_targets, test_predictions, alpha=0.6)
plt.plot(
    [np.min(test_targets), np.max(test_targets)],
    [np.min(test_targets), np.max(test_targets)],
    "r--",
)
plt.xlabel("Actual Chl Values")
plt.ylabel("Predicted Chl Values")
plt.title(
    f"CNN Regression with Augmentation: Predictions vs Actual Values\nR² = {r2_score(test_targets, test_predictions):.4f}"
)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(
    str(project_root) + "/plots/cnn_regression_augmented_predictions.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Compare results between models
print("\n" + "=" * 80)
print("COMPARISON BETWEEN MODELS (WITH VS WITHOUT AUGMENTATION)")
print("=" * 80)

# Compare the two models by evaluating both on the test set
model.eval()
model_aug.eval()

with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_reg).unsqueeze(1)

    # Get predictions from both models
    test_pred_standard = model(X_test_tensor).detach().numpy()
    test_pred_augmented = model_aug(X_test_tensor).detach().numpy()

    # Convert back to original scale
    test_pred_standard = scaler_y_Chl.inverse_transform(test_pred_standard)
    test_pred_augmented = scaler_y_Chl.inverse_transform(test_pred_augmented)
    test_targets = scaler_y_Chl.inverse_transform(y_test_chl)

# Calculate metrics for both models
r2_standard = r2_score(test_targets, test_pred_standard)
r2_augmented = r2_score(test_targets, test_pred_augmented)

mse_standard = mean_squared_error(test_targets, test_pred_standard)
mse_augmented = mean_squared_error(test_targets, test_pred_augmented)

rmse_standard = np.sqrt(mse_standard)
rmse_augmented = np.sqrt(mse_augmented)

mae_standard = mean_absolute_error(test_targets, test_pred_standard)
mae_augmented = mean_absolute_error(test_targets, test_pred_augmented)

# Print comparison
print(f"R² score (standard): {r2_standard:.4f}")
print(f"R² score (augmented): {r2_augmented:.4f}")
print(f"Improvement: {(r2_augmented - r2_standard) / r2_standard * 100:.2f}%")
print()

print(f"MSE (standard): {mse_standard:.4f}")
print(f"MSE (augmented): {mse_augmented:.4f}")
print(f"Improvement: {(mse_standard - mse_augmented) / mse_standard * 100:.2f}%")
print()

print(f"RMSE (standard): {rmse_standard:.4f}")
print(f"RMSE (augmented): {rmse_augmented:.4f}")
print(f"Improvement: {(rmse_standard - rmse_augmented) / rmse_standard * 100:.2f}%")
print()

print(f"MAE (standard): {mae_standard:.4f}")
print(f"MAE (augmented): {mae_augmented:.4f}")
print(f"Improvement: {(mae_standard - mae_augmented) / mae_standard * 100:.2f}%")

# Visualize predictions from both models
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(test_targets, test_pred_standard, alpha=0.6, c="blue")
plt.plot(
    [np.min(test_targets), np.max(test_targets)],
    [np.min(test_targets), np.max(test_targets)],
    "r--",
)
plt.xlabel("Actual Chl Values")
plt.ylabel("Predicted Chl Values")
plt.title(f"Standard Model\nR² = {r2_standard:.4f}")
plt.grid(True, linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
plt.scatter(test_targets, test_pred_augmented, alpha=0.6, c="green")
plt.plot(
    [np.min(test_targets), np.max(test_targets)],
    [np.min(test_targets), np.max(test_targets)],
    "r--",
)
plt.xlabel("Actual Chl Values")
plt.ylabel("Predicted Chl Values")
plt.title(f"Augmented Model\nR² = {r2_augmented:.4f}")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig(
    str(project_root) + "/plots/cnn_regression_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
