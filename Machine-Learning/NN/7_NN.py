import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler  # ajout pour l'oversampling

# ------------------------------
# 0. Définition des hyperparamètres et options
# ------------------------------
hp = {
    "lr": 0.005,             # mise à jour : learning rate
    "batch_size": 64,        # mise à jour : taille de lot
    "num_epochs": 100,       # inchangé
    "dropout_rate": 0.0,     # mise à jour : pas de dropout
    "weight_decay": 0.01     # mise à jour : régularisation L2
}
apply_pretreatment = True  # Flag pour activer le prétraitement (SNV)

def snv(X):
    """Application de la transformation SNV (ligne par ligne)."""
    X_array = X.values
    X_snv = np.array([ (row - np.mean(row)) / (np.std(row) if np.std(row)!=0 else 1) for row in X_array])
    return pd.DataFrame(X_snv, columns=X.columns)

# Ajout: Transformation par première dérivée (le pretreatment le plus intéressant)
def first_derivative(X):
    """Application de la première dérivée (ligne par ligne) via np.gradient."""
    X_array = X.values
    X_deriv = np.array([np.gradient(row) for row in X_array])
    return pd.DataFrame(X_deriv, columns=X.columns)

# Ajout: Fonction utilitaire pour l'oversampling
def oversample_data(X, y):
    ros = RandomOverSampler(random_state=42)
    return ros.fit_resample(X, y)

# ------------------------------
# 1. Chargement et préparation des données
# ------------------------------
data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)
target_col = data.columns[-1]

# Encodage de la cible
le = LabelEncoder()
data[target_col] = le.fit_transform(data[target_col])
n_classes = data[target_col].nunique()

# Création du jeu de validation : 6 échantillons par classe
validation_frames = []
for cls in data[target_col].unique():
    cls_df = data[data[target_col] == cls]
    if len(cls_df) >= 6:
        val_cls = cls_df.sample(n=6, random_state=42)
    else:
        val_cls = cls_df.sample(n=6, replace=True, random_state=42)
    validation_frames.append(val_cls)
validation_data = pd.concat(validation_frames)
training_data = data.drop(validation_data.index)

# Séparation en features et cible
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]
X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# ------------------------------
# 1.a Prétraitement (optionnel)
# ------------------------------
if apply_pretreatment:
    # Utilisation de la première dérivée en remplacement du SNV
    X_train = first_derivative(X_train)
    X_val = first_derivative(X_val)

# ------------------------------
# 1.b Oversampling sur le jeu d’entraînement
# ------------------------------
# Remplacement de l'oversampling direct par l'appel à la fonction oversample_data
X_train_over, y_train_over = oversample_data(X_train, y_train)

# ------------------------------
# 1.c Standardisation
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_over)
X_val_scaled = scaler.transform(X_val)

# ------------------------------
# 2. Préparation des tenseurs pour PyTorch
# ------------------------------
# Pour un CNN 1D, on ajoute une dimension canal
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_over.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
num_features = X_train_tensor.shape[2]

# ------------------------------
# 3. Définition du modèle CNN (inspiré des scripts de préparation et de model_validation.py) avec hyperparamètre dropout
# ------------------------------
class CNN(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Réduit à (batch, 32, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x: (batch_size, 1, num_features)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Aplatissement
        x = self.dropout(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_features=num_features, num_classes=n_classes, dropout_rate=hp["dropout_rate"])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

# ------------------------------
# 4. Boucle d'entraînement et suivi de la loss
# ------------------------------
train_losses = []
val_losses = []

for epoch in range(hp["num_epochs"]):
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
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{hp['num_epochs']} - Training Loss: {epoch_train_loss:.4f} - Validation Loss: {loss_val:.4f}")

# ------------------------------
# 5. Évaluation sur le jeu de validation
# ------------------------------
model.eval()
with torch.no_grad():
    outputs_val = model(X_val_tensor.to(device))
    _, y_pred_tensor = torch.max(outputs_val, 1)
    y_pred = y_pred_tensor.cpu().numpy()

print("\n=== Validation Simple ===")
print("Classification Report:\n", classification_report(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Validation Simple)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()
print("Score de Validation Simple :", accuracy_score(y_val, y_pred))

# ------------------------------
# 6. Affichage des courbes d'apprentissage
# ------------------------------
epochs_arr = np.arange(1, hp["num_epochs"]+1)
plt.figure(figsize=(10,6))
plt.plot(epochs_arr, train_losses, marker='o', label='Training Loss')
plt.plot(epochs_arr, val_losses, marker='s', label='Validation Loss')
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.title("Courbes d'entraînement et validation")
plt.legend()
plt.tight_layout()
plt.show()
