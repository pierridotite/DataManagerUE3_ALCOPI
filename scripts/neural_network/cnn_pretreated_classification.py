"""Convolutional Neural Network for classification of num_classe with preprocessing.

This script implements a CNN for spectral data classification with various
preprocessing options, focusing on preprocessing the spectral data before
training the neural network.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time  # Import for measuring training time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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
LEARNING_RATE = 0.0001
# Proportion of data for the test set
TEST_SIZE = 0.25
# Proportion of data for the validation set (relative to remaining data)
VAL_SIZE = 0.2
# Kernel size for convolutional layers
KERNEL_SIZE = 10
# Stride for convolutional layers
STRIDE = 1
# Padding for convolutional layers
PADDING = 2
# ==============================================================================

# Get current notebook path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataLoading import (
    data_3cl,
    spectral_cols,
    mat_snv_sg_3cl,
    mat_deriv1_3cl,
    mat_deriv2_3cl,
)
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


def print_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, set_name: str = "test"
) -> None:
    """
    Print standard classification metrics

    Args:
        y_true: Actual class labels
        y_pred: Model predicted class labels
        set_name: Name of the dataset for reporting (e.g., 'train', 'test')
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy on {set_name} set: {accuracy:.4f}")

    # Calculate precision, recall, and F1 score (macro avg for multiclass)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Precision on {set_name} set: {precision:.4f}")
    print(f"Recall on {set_name} set: {recall:.4f}")
    print(f"F1 score on {set_name} set: {f1:.4f}")

    # Generate and print classification report
    print("\nClassification Report:")
    cls_report = classification_report(y_true, y_pred, digits=4)
    print(cls_report)

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({set_name} set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


# Custom Dataset class for spectral data with multiple transformations
class MultiSpectralDataset(Dataset):
    """Dataset class to handle multiple spectral data transformations for PyTorch"""

    def __init__(
        self,
        X_original: np.ndarray | spmatrix,
        X_snv_sg: np.ndarray | spmatrix,
        X_deriv1: np.ndarray | spmatrix,
        X_deriv2: np.ndarray | spmatrix,
        y: np.ndarray | spmatrix,
    ):
        # Reshape each X for CNN input: [batch_size, channels, sequence_length]
        self.X_original = torch.FloatTensor(X_original).unsqueeze(
            1
        )  # Add channel dimension
        self.X_snv_sg = torch.FloatTensor(X_snv_sg).unsqueeze(1)
        self.X_deriv1 = torch.FloatTensor(X_deriv1).unsqueeze(1)
        self.X_deriv2 = torch.FloatTensor(X_deriv2).unsqueeze(1)
        self.y = torch.LongTensor(y)  # LongTensor for classification targets

    def __len__(self) -> int:
        return len(self.X_original)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        return [
            self.X_original[idx],
            self.X_snv_sg[idx],
            self.X_deriv1[idx],
            self.X_deriv2[idx],
        ], self.y[idx]


# CNN model with multiple inputs for classification
class MultiInputCNNModel(nn.Module):
    """CNN model with parallel paths for multiple spectral transformations

    Args:
        input_dim: Length of the spectral sequence
        output_dim: Number of classes for classification
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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

        # Define CNN path for original spectral data
        self.cnn_original = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
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

        # Define CNN path for SNV + Savitzky-Golay transformed data
        self.cnn_snv_sg = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
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

        # Define CNN path for first derivative transformed data
        self.cnn_deriv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
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

        # Define CNN path for second derivative transformed data
        self.cnn_deriv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
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

        # Calculate total features after concatenation
        total_features = 32 * pool2_output_size * 4  # 4 paths

        # Fully connected layers that combine all paths for classification
        self.fc = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        # Process each input through its respective CNN path
        x_original = self.cnn_original(x_list[0])
        x_snv_sg = self.cnn_snv_sg(x_list[1])
        x_deriv1 = self.cnn_deriv1(x_list[2])
        x_deriv2 = self.cnn_deriv2(x_list[3])

        # Flatten each output
        x_original = x_original.view(x_original.size(0), -1)
        x_snv_sg = x_snv_sg.view(x_snv_sg.size(0), -1)
        x_deriv1 = x_deriv1.view(x_deriv1.size(0), -1)
        x_deriv2 = x_deriv2.view(x_deriv2.size(0), -1)

        # Concatenate all flattened outputs
        x_combined = torch.cat([x_original, x_snv_sg, x_deriv1, x_deriv2], dim=1)

        # Pass through fully connected layers
        return self.fc(x_combined)


# =============================================================================
# CNN for classification with multiple pretreated spectral inputs
# =============================================================================
print("\n" + "=" * 80)
print("CNN FOR CLASSIFICATION (NUM_CLASSE PREDICTION) WITH MULTIPLE PRETREATED INPUTS")
print("=" * 80)

# Preparing data for classification
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

# Encode target variable (num_classe)
label_encoder = LabelEncoder()
all_classes = np.concatenate(
    [
        train_data["num_classe"].values,
        val_data["num_classe"].values,
        test_data["num_classe"].values,
    ]
)
label_encoder.fit(all_classes)

# Get the number of unique classes for model output dimension
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# Prepare features and targets for different sets
# Get original spectral data
X_train_original = np.array(train_data[spectral_cols])
X_val_original = np.array(val_data[spectral_cols])
X_test_original = np.array(test_data[spectral_cols])

# Get SNV+SG transformed data
X_train_snv_sg = np.array(mat_snv_sg_3cl.iloc[indices_train][spectral_cols])
X_val_snv_sg = np.array(mat_snv_sg_3cl.iloc[indices_val][spectral_cols])
X_test_snv_sg = np.array(mat_snv_sg_3cl.iloc[indices_test][spectral_cols])

# Get first derivative transformed data
X_train_deriv1 = np.array(mat_deriv1_3cl.iloc[indices_train][spectral_cols])
X_val_deriv1 = np.array(mat_deriv1_3cl.iloc[indices_val][spectral_cols])
X_test_deriv1 = np.array(mat_deriv1_3cl.iloc[indices_test][spectral_cols])

# Get second derivative transformed data
X_train_deriv2 = np.array(mat_deriv2_3cl.iloc[indices_train][spectral_cols])
X_val_deriv2 = np.array(mat_deriv2_3cl.iloc[indices_val][spectral_cols])
X_test_deriv2 = np.array(mat_deriv2_3cl.iloc[indices_test][spectral_cols])

# Get target variables (num_classe) and encode them
y_train_cls = label_encoder.transform(train_data["num_classe"])
y_val_cls = label_encoder.transform(val_data["num_classe"])
y_test_cls = label_encoder.transform(test_data["num_classe"])

# Standardize each spectral dataset
scaler_original = StandardScaler()
scaler_snv_sg = StandardScaler()
scaler_deriv1 = StandardScaler()
scaler_deriv2 = StandardScaler()

# Scale original spectral data
X_train_original = scaler_original.fit_transform(X_train_original)
X_val_original = scaler_original.transform(X_val_original)
X_test_original = scaler_original.transform(X_test_original)

# Scale SNV+SG transformed data
X_train_snv_sg = scaler_snv_sg.fit_transform(X_train_snv_sg)
X_val_snv_sg = scaler_snv_sg.transform(X_val_snv_sg)
X_test_snv_sg = scaler_snv_sg.transform(X_test_snv_sg)

# Scale first derivative transformed data
X_train_deriv1 = scaler_deriv1.fit_transform(X_train_deriv1)
X_val_deriv1 = scaler_deriv1.transform(X_val_deriv1)
X_test_deriv1 = scaler_deriv1.transform(X_test_deriv1)

# Scale second derivative transformed data
X_train_deriv2 = scaler_deriv2.fit_transform(X_train_deriv2)
X_val_deriv2 = scaler_deriv2.transform(X_val_deriv2)
X_test_deriv2 = scaler_deriv2.transform(X_test_deriv2)

# Create datasets and loaders for training and validation
train_dataset = MultiSpectralDataset(
    X_train_original, X_train_snv_sg, X_train_deriv1, X_train_deriv2, y_train_cls
)
val_dataset = MultiSpectralDataset(
    X_val_original, X_val_snv_sg, X_val_deriv1, X_val_deriv2, y_val_cls
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function and optimizer
model = MultiInputCNNModel(input_dim=len(spectral_cols), output_dim=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics
metrics_history: List[TrainingMetrics] = []

print("Starting CNN training for classification with preprocessing...")
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
    if val_loss < best_val_loss:
        best_val_loss = val_loss
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
    print(
        f"Best model loaded with validation loss of {best_val_loss/len(val_loader):.4f}"
    )

# Evaluation on test set
model.eval()
with torch.no_grad():
    # Create dataset for test data
    test_inputs = [
        torch.FloatTensor(X_test_original).unsqueeze(1),
        torch.FloatTensor(X_test_snv_sg).unsqueeze(1),
        torch.FloatTensor(X_test_deriv1).unsqueeze(1),
        torch.FloatTensor(X_test_deriv2).unsqueeze(1),
    ]

    # Get predictions for all test data at once
    test_outputs = model(test_inputs)
    _, test_predictions = torch.max(test_outputs, 1)
    test_predictions = test_predictions.cpu().numpy()

# Print evaluation metrics
print_classification_metrics(y_test_cls, test_predictions, "test")

# Visualize the evolution of metrics
plot_training_metrics(
    metrics_history,
    title="Evolution of Training Metrics for CNN Classification (Multiple Pretreated Inputs)",
    save_path=str(project_root)
    + "/plots/training_metrics_cnn_pretreated_classification.png",
)

# =============================================================================
# CNN for classification with multiple pretreated spectral inputs and data augmentation
# =============================================================================
print("\n" + "=" * 80)
print(
    "CNN FOR CLASSIFICATION (NUM_CLASSE PREDICTION) WITH MULTIPLE PRETREATED INPUTS AND DATA AUGMENTATION"
)
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

# Prepare original spectral features (augmented)
X_train_original = np.array(train_data_augmented[spectral_cols])
X_val_original = np.array(val_data[spectral_cols])
X_test_original = np.array(test_data[spectral_cols])

# Get target variables (num_classe - augmented) and encode them
y_train_cls = label_encoder.transform(train_data_augmented["num_classe"])
y_val_cls = label_encoder.transform(val_data["num_classe"])
y_test_cls = label_encoder.transform(test_data["num_classe"])

# We need to re-apply transformations on the augmented data
# First, get the transformers from dataLoading module
from scripts.dataLoading import (
    snv_transformer,
    sg_smoother,
    derivative1,
    derivative2,
)

# Apply transformations on augmented data
# SNV + SG transformation
X_train_snv_sg = sg_smoother.transform(snv_transformer.transform(X_train_original))
# First derivative transformation
X_train_deriv1 = derivative1.transform(X_train_original)
# Second derivative transformation
X_train_deriv2 = derivative2.transform(X_train_original)

# For validation and test, we keep the existing transformed data
X_val_snv_sg = np.array(mat_snv_sg_3cl.iloc[indices_val][spectral_cols])
X_test_snv_sg = np.array(mat_snv_sg_3cl.iloc[indices_test][spectral_cols])

X_val_deriv1 = np.array(mat_deriv1_3cl.iloc[indices_val][spectral_cols])
X_test_deriv1 = np.array(mat_deriv1_3cl.iloc[indices_test][spectral_cols])

X_val_deriv2 = np.array(mat_deriv2_3cl.iloc[indices_val][spectral_cols])
X_test_deriv2 = np.array(mat_deriv2_3cl.iloc[indices_test][spectral_cols])

# Standardize each spectral dataset
scaler_original_aug = StandardScaler()
scaler_snv_sg_aug = StandardScaler()
scaler_deriv1_aug = StandardScaler()
scaler_deriv2_aug = StandardScaler()

# Scale original spectral data
X_train_original = scaler_original_aug.fit_transform(X_train_original)
X_val_original = scaler_original_aug.transform(X_val_original)
X_test_original = scaler_original_aug.transform(X_test_original)

# Scale SNV+SG transformed data
X_train_snv_sg = scaler_snv_sg_aug.fit_transform(X_train_snv_sg)
X_val_snv_sg = scaler_snv_sg_aug.transform(X_val_snv_sg)
X_test_snv_sg = scaler_snv_sg_aug.transform(X_test_snv_sg)

# Scale first derivative transformed data
X_train_deriv1 = scaler_deriv1_aug.fit_transform(X_train_deriv1)
X_val_deriv1 = scaler_deriv1_aug.transform(X_val_deriv1)
X_test_deriv1 = scaler_deriv1_aug.transform(X_test_deriv1)

# Scale second derivative transformed data
X_train_deriv2 = scaler_deriv2_aug.fit_transform(X_train_deriv2)
X_val_deriv2 = scaler_deriv2_aug.transform(X_val_deriv2)
X_test_deriv2 = scaler_deriv2_aug.transform(X_test_deriv2)

# Create datasets and loaders for training and validation
train_dataset = MultiSpectralDataset(
    X_train_original, X_train_snv_sg, X_train_deriv1, X_train_deriv2, y_train_cls
)
val_dataset = MultiSpectralDataset(
    X_val_original, X_val_snv_sg, X_val_deriv1, X_val_deriv2, y_val_cls
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function and optimizer
model_aug = MultiInputCNNModel(input_dim=len(spectral_cols), output_dim=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_aug.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics for augmented model
metrics_history: List[TrainingMetrics] = []

print(
    "Starting CNN training with multiple pretreated inputs and data augmentation for classification..."
)
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
    if val_loss < best_val_loss:
        best_val_loss = val_loss
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
    print(
        f"Best model loaded with validation loss of {best_val_loss/len(val_loader):.4f}"
    )

# Evaluation on test set
model_aug.eval()
with torch.no_grad():
    # Create dataset for test data
    test_inputs = [
        torch.FloatTensor(X_test_original).unsqueeze(1),
        torch.FloatTensor(X_test_snv_sg).unsqueeze(1),
        torch.FloatTensor(X_test_deriv1).unsqueeze(1),
        torch.FloatTensor(X_test_deriv2).unsqueeze(1),
    ]

    # Get predictions for all test data at once
    test_outputs = model_aug(test_inputs)
    _, test_predictions = torch.max(test_outputs, 1)
    test_predictions = test_predictions.cpu().numpy()

# Print evaluation metrics
print_classification_metrics(y_test_cls, test_predictions, "test")

# Visualize the evolution of metrics
plot_training_metrics(
    metrics_history,
    title="Evolution of Training Metrics for CNN Classification (Multiple Pretreated Inputs with Augmentation)",
    save_path=str(project_root)
    + "/plots/training_metrics_cnn_pretreated_classification_augmented.png",
)
