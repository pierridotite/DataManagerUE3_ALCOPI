import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.transformers import (
    SNVTransformer,
    SavitzkyGolayTransformer,
    DerivativeTransformer,
)

# Define data paths
DATA_PATH = project_root / "data"

# Read data for 3 classes dataset
mat_3cl = pd.read_csv(DATA_PATH  / "spectra_3cl.csv", sep=";", index_col=0)
simple_classes_3cl = pd.read_csv(
    DATA_PATH  / "classes_3cl.csv", sep=";", index_col=0
)

# Read leaves data
dt_leaves = pd.read_csv(DATA_PATH / "dt_leaves.csv", sep=";")

# Convert categorical columns
categorical_columns = ["variety", "symptom", "plotLocation"]
for col in categorical_columns:
    dt_leaves[col] = dt_leaves[col].astype("category")

spectral_axis = [float(i) for i in mat_3cl.columns.tolist()]

# Create and merge data for 3 classes
data_3cl = simple_classes_3cl.merge(
    mat_3cl, left_index=True, right_index=True
)  # Merge classes and spectral data
data_3cl = data_3cl.merge(
    dt_leaves, left_index=True, right_on="directoryName"
)  # Add categorical variables using directoryName as key
data_3cl = data_3cl.drop(
    ["directoryName", "imageID"], axis=1
)  # Remove directoryName column

# Convert all categorical columns
categorical_columns.extend(["num_classe", "biotic", "abiotic", "healthy"])
for col in categorical_columns:
    data_3cl[col] = data_3cl[col].astype("category")

# Apply spectral transformations on mat_3cl
spectral_cols = mat_3cl.columns  # All columns in mat_3cl are spectral
X_spectral = mat_3cl.to_numpy()

# Initialize transformers
snv_transformer = SNVTransformer()
sg_smoother = SavitzkyGolayTransformer(window_length=11, polyorder=3)
derivative1 = DerivativeTransformer(order=1, window_length=15, polyorder=3)
derivative2 = DerivativeTransformer(order=2, window_length=21, polyorder=3)

# Create transformed matrices
mat_snv_3cl = pd.DataFrame(
    snv_transformer.fit_transform(X_spectral),
    columns=spectral_cols,
    index=data_3cl.index
)
mat_sg_3cl = pd.DataFrame(
    sg_smoother.fit_transform(X_spectral),
    columns=spectral_cols,
    index=data_3cl.index
)
mat_deriv1_3cl = pd.DataFrame(
    derivative1.fit_transform(X_spectral),
    columns=spectral_cols,
    index=data_3cl.index
)
mat_deriv2_3cl = pd.DataFrame(
    derivative2.fit_transform(X_spectral),
    columns=spectral_cols,
    index=data_3cl.index
)
mat_snv_sg_3cl = pd.DataFrame(
    sg_smoother.fit_transform(snv_transformer.fit_transform(X_spectral)),
    columns=spectral_cols,
    index=data_3cl.index
)
mat_snv_deriv1_3cl = pd.DataFrame(
    derivative1.fit_transform(snv_transformer.fit_transform(X_spectral)),
    columns=spectral_cols,
    index=data_3cl.index
)
