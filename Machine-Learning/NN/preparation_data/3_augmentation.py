import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid, StratifiedKFold

# Pour fixer la graine dans la fonction d'augmentation (optionnel)
np.random.seed(42)
torch.manual_seed(42)

# ================================
# 1. Chargement et préparation des données
# ================================
data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

# (Optionnel) Encodage de la cible si nécessaire
if data[target_col].dtype == 'object':
    le = LabelEncoder()
    data[target_col] = le.fit_transform(data[target_col])

# Création du jeu de validation : 6 individus par espèce
validation_frames = []
for specie in data[target_col].unique():
    specie_df = data[data[target_col] == specie]
    if len(specie_df) >= 6:
        val_specie = specie_df.sample(n=6, random_state=42)
    else:
        val_specie = specie_df.sample(n=6, replace=True, random_state=42)
    validation_frames.append(val_specie)
validation_data = pd.concat(validation_frames)

# Le jeu d'entraînement correspond aux données restantes
training_data = data.drop(validation_data.index)

# Séparation en features et target
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# ================================
# 2. Définition de la fonction d'augmentation
# ================================
def augment_spectrum(spectrum, scale_min, scale_max, noise_std, shift_min, shift_max):
    """
    Augmente un spectre en appliquant :
      - une mise à l'échelle aléatoire,
      - l'ajout de bruit gaussien,
      - un décalage spectral.
    """
    # Mise à l'échelle aléatoire
    scale_factor = np.random.uniform(scale_min, scale_max)
    spectrum_aug = spectrum * scale_factor

    # Ajout de bruit gaussien
    noise = np.random.normal(0, noise_std, size=spectrum.shape)
    spectrum_aug += noise

    # Décalage spectral (rotation des indices)
    shift = np.random.randint(shift_min, shift_max + 1)
    spectrum_aug = np.roll(spectrum_aug, shift)
    
    return spectrum_aug

# ================================
# 3. Définition du modèle CNN (PyTorch)
# ================================
class CNN(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Réduit la dimension à (batch, 32, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x : (batch_size, 1, num_features)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # aplatissement
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Paramètres d'entraînement pour les essais dans la grid search
batch_size_grid = 32
num_epochs_grid = 10     # nombre d'époques réduit pour accélérer la grid search
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nombre fixe d'exemples augmentés par classe
n_aug_per_class = 10

# ================================
# 4. Grid Search sur les paramètres d'augmentation
# ================================
param_grid_aug = {
    'scale_min': [0.80, 0.85],
    'scale_max': [1.15, 1.20],
    'noise_std': [0.005, 0.01, 0.015],
    'shift_min': [-8, -5, -3],
    'shift_max': [3, 5, 8]
}

results = []
grid = list(ParameterGrid(param_grid_aug))
print("Début de la grid search sur {} combinaisons...".format(len(grid)))

for params in grid:
    augmented_X_list = []
    augmented_y_list = []
    
    # Pour chaque classe, générer exactement n_aug_per_class exemples augmentés
    for cls in y_train.unique():
        cls_indices = X_train[y_train == cls].index
        chosen_indices = np.random.choice(cls_indices, size=n_aug_per_class, replace=True)
        for idx in chosen_indices:
            spectrum = X_train.loc[idx].values.astype(float)
            aug_spec = augment_spectrum(
                spectrum,
                params['scale_min'],
                params['scale_max'],
                params['noise_std'],
                params['shift_min'],
                params['shift_max']
            )
            augmented_X_list.append(aug_spec)
            augmented_y_list.append(cls)
    
    # Reconstruction des DataFrames pour les exemples augmentés
    X_train_aug = pd.DataFrame(augmented_X_list, columns=X_train.columns)
    y_train_aug = pd.Series(augmented_y_list, name=target_col)
    
    # Combinaison des données originales et augmentées
    X_train_final = pd.concat([X_train, X_train_aug], ignore_index=True)
    y_train_final = pd.concat([y_train, y_train_aug], ignore_index=True)
    
    # Préparation des tenseurs pour le CNN
    X_train_tensor = torch.tensor(X_train_final.values, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_final.values, dtype=torch.long)
    
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    
    # Création du DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_grid, shuffle=True)
    
    # Définir le modèle CNN
    num_features = X_train_tensor.shape[2]
    num_classes = y_train_tensor.unique().numel()
    model = CNN(num_features=num_features, num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entraînement sur num_epochs_grid époques
    model.train()
    for epoch in range(num_epochs_grid):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        # (On peut afficher la loss par époque si besoin)
    
    # Évaluation sur le jeu de validation
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_tensor.to(device))
        _, y_pred_tensor = torch.max(outputs_val, 1)
        y_pred = y_pred_tensor.cpu().numpy()
    acc = accuracy_score(y_val, y_pred)
    results.append({'params': params, 'accuracy': acc})
    print("Params:", params, "-> Accuracy:", acc)

# Transformation des résultats en DataFrame et sélection de la meilleure configuration
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='accuracy', ascending=False)
best_params = results_df_sorted.iloc[0]['params']
best_accuracy = results_df_sorted.iloc[0]['accuracy']

print("\nMeilleure configuration d'augmentation :")
print(best_params)
print("Accuracy :", best_accuracy)

# ================================
# 5. Visualisation d'un exemple d'augmentation avec la meilleure configuration
# ================================
sample_idx = X_train.index[0]
original_spectrum = X_train.loc[sample_idx].values.astype(float)
augmented_spectrum = augment_spectrum(
    original_spectrum,
    best_params['scale_min'],
    best_params['scale_max'],
    best_params['noise_std'],
    best_params['shift_min'],
    best_params['shift_max']
)

plt.figure(figsize=(10, 5))
plt.plot(original_spectrum, label='Original')
plt.plot(augmented_spectrum, label='Augmenté', linestyle='--')
plt.title("Comparaison d'un spectre original et de sa version augmentée")
plt.xlabel("Index du spectre")
plt.ylabel("Intensité")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 6. Entraînement final avec la meilleure configuration d'augmentation
# ================================
augmented_X_list = []
augmented_y_list = []

for cls in y_train.unique():
    cls_indices = X_train[y_train == cls].index
    chosen_indices = np.random.choice(cls_indices, size=n_aug_per_class, replace=True)
    for idx in chosen_indices:
        spectrum = X_train.loc[idx].values.astype(float)
        aug_spec = augment_spectrum(
            spectrum,
            best_params['scale_min'],
            best_params['scale_max'],
            best_params['noise_std'],
            best_params['shift_min'],
            best_params['shift_max']
        )
        augmented_X_list.append(aug_spec)
        augmented_y_list.append(cls)

X_train_aug = pd.DataFrame(augmented_X_list, columns=X_train.columns)
y_train_aug = pd.Series(augmented_y_list, name=target_col)

X_train_final = pd.concat([X_train, X_train_aug], ignore_index=True)
y_train_final = pd.concat([y_train, y_train_aug], ignore_index=True)

# Préparation des tenseurs pour l'entraînement final
X_train_tensor = torch.tensor(X_train_final.values, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_final.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_features = X_train_tensor.shape[2]
num_classes = y_train_tensor.unique().numel()
model = CNN(num_features=num_features, num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs_final = 50  # Nombre d'époques pour l'entraînement final

train_losses = []
val_losses = []

for epoch in range(num_epochs_final):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_tensor.to(device))
        loss_val = criterion(outputs_val, y_val_tensor.to(device)).item()
    val_losses.append(loss_val)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs_final} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {loss_val:.4f}")

# Évaluation finale sur le jeu de validation
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor.to(device))
    _, y_pred_tensor = torch.max(outputs, 1)
    y_pred = y_pred_tensor.cpu().numpy()
    
print("\n=== Validation Simple avec CNN ===")
print("Classification Report:\n", classification_report(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Matrice de confusion (Validation Simple)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()
print("Accuracy finale :", accuracy_score(y_val, y_pred))

# ================================
# 7. Validation croisée avec augmentation (sur l'ensemble complet)
# ================================
X_full = data.drop(columns=[target_col])
y_full = data[target_col]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

for train_idx, test_idx in cv.split(X_full, y_full):
    X_train_cv = X_full.iloc[train_idx].reset_index(drop=True)
    y_train_cv = y_full.iloc[train_idx].reset_index(drop=True)
    X_test_cv = X_full.iloc[test_idx].reset_index(drop=True)
    y_test_cv = y_full.iloc[test_idx].reset_index(drop=True)
    
    # Augmentation sur le pli d'entraînement
    augmented_X_list = []
    augmented_y_list = []
    for cls in y_train_cv.unique():
        cls_indices = X_train_cv[y_train_cv == cls].index
        chosen_indices = np.random.choice(cls_indices, size=n_aug_per_class, replace=True)
        for idx in chosen_indices:
            spectrum = X_train_cv.loc[idx].values.astype(float)
            aug_spec = augment_spectrum(
                spectrum,
                best_params['scale_min'],
                best_params['scale_max'],
                best_params['noise_std'],
                best_params['shift_min'],
                best_params['shift_max']
            )
            augmented_X_list.append(aug_spec)
            augmented_y_list.append(cls)
    
    if augmented_X_list:
        X_train_cv_aug = pd.DataFrame(augmented_X_list, columns=X_train_cv.columns)
        y_train_cv_aug = pd.Series(augmented_y_list, name=target_col)
        X_train_final_cv = pd.concat([X_train_cv, X_train_cv_aug], ignore_index=True)
        y_train_final_cv = pd.concat([y_train_cv, y_train_cv_aug], ignore_index=True)
    else:
        X_train_final_cv = X_train_cv.copy()
        y_train_final_cv = y_train_cv.copy()
    
    # Préparation des tenseurs pour le pli courant
    X_train_cv_tensor = torch.tensor(X_train_final_cv.values, dtype=torch.float32).unsqueeze(1)
    y_train_cv_tensor = torch.tensor(y_train_final_cv.values, dtype=torch.long)
    X_test_cv_tensor = torch.tensor(X_test_cv.values, dtype=torch.float32).unsqueeze(1)
    y_test_cv_tensor = torch.tensor(y_test_cv.values, dtype=torch.long)
    
    train_dataset_cv = TensorDataset(X_train_cv_tensor, y_train_cv_tensor)
    train_loader_cv = DataLoader(train_dataset_cv, batch_size=32, shuffle=True)
    
    num_features_cv = X_train_cv_tensor.shape[2]
    num_classes_cv = y_train_cv_tensor.unique().numel()
    model_cv = CNN(num_features=num_features_cv, num_classes=num_classes_cv)
    model_cv.to(device)
    
    criterion_cv = nn.CrossEntropyLoss()
    optimizer_cv = optim.Adam(model_cv.parameters(), lr=learning_rate)
    
    num_epochs_cv = 30  # nombre d'époques réduit pour la CV
    model_cv.train()
    for epoch in range(num_epochs_cv):
        for batch_X, batch_y in train_loader_cv:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer_cv.zero_grad()
            outputs_cv = model_cv(batch_X)
            loss_cv = criterion_cv(outputs_cv, batch_y)
            loss_cv.backward()
            optimizer_cv.step()
    
    # Évaluation sur le pli test
    model_cv.eval()
    with torch.no_grad():
        outputs_cv = model_cv(X_test_cv_tensor.to(device))
        _, y_pred_cv_tensor = torch.max(outputs_cv, 1)
        y_pred_cv = y_pred_cv_tensor.cpu().numpy()
    acc_cv = accuracy_score(y_test_cv, y_pred_cv)
    cv_accuracies.append(acc_cv)

print("\nAccuracies en validation croisée (CNN avec augmentation) :", cv_accuracies)
print("Accuracy moyenne en validation croisée :", np.mean(cv_accuracies))
