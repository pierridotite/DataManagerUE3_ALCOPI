import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# ================================
# 1. Chargement et préparation des données
# ================================

data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

# Encodage de la cible si nécessaire
if data[target_col].dtype == 'object':
    le = LabelEncoder()
    data[target_col] = le.fit_transform(data[target_col])

# Création du jeu de validation : 6 individus par espèce
validation_frames = []
for specie in data[target_col].unique():
    specie_df = data[data[target_col] == specie]
    # Prélèvement sans remise si possible, sinon avec remise
    if len(specie_df) >= 6:
        val_specie = specie_df.sample(n=6, random_state=42)
    else:
        val_specie = specie_df.sample(n=6, replace=True, random_state=42)
    validation_frames.append(val_specie)
validation_data = pd.concat(validation_frames)

# Le jeu d’entraînement correspond aux données restantes
training_data = data.drop(validation_data.index)

# Séparation en features et target pour l’entraînement et la validation
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# Application de l'oversampling sur le jeu d’entraînement
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
# Comptage des effectifs après oversampling (pour affichage dans l'histogramme)
oversampled_counts = pd.Series(y_train_over).value_counts().sort_index()

# ================================
# 2. Histogramme des effectifs par espèce
# ================================

# Comptes dans le jeu complet (pour référence)
full_counts = data[target_col].value_counts().sort_index()
# Comptes dans le jeu d'entraînement et de validation
train_counts = training_data[target_col].value_counts().sort_index()
val_counts = validation_data[target_col].value_counts().sort_index()

# On trie les espèces selon les effectifs décroissants dans le jeu d'entraînement
order = train_counts.sort_values(ascending=False).index
train_counts = train_counts.reindex(order)
val_counts = val_counts.reindex(order, fill_value=0)
oversampled_counts = oversampled_counts.reindex(order, fill_value=0)
additional_counts = oversampled_counts - train_counts

species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
plt.bar(x, val_counts, width, color='green', label='Validation dataset (6 per species)')
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training dataset')
plt.bar(x, additional_counts, width, bottom=val_counts + train_counts, color='orange',
        label='Oversampled training data')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme : Effectifs par espèce (Validation + Training + Oversampling)")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 3. Définition du modèle CNN et validation simple (training/validation)
# ================================

# Pour utiliser un CNN sur des données tabulaires, on considère chaque vecteur
# de features comme une "séquence" avec un seul canal. On ajoute donc une dimension :
# reshape en (n_samples, 1, n_features).

# Définition d'un modèle CNN simple
class CNN(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Pooling global pour réduire la dimension à (batch, 32, 1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
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

# Paramètres d'entraînement
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Conversion des données oversamplées en tenseurs PyTorch
# Utilisation de .values pour convertir les DataFrame en array NumPy.
# On ajoute .unsqueeze(1) pour obtenir un tenseur de forme (n_samples, 1, n_features)
X_train_tensor = torch.tensor(X_train_over.values, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(np.array(y_train_over), dtype=torch.long)

X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Création du DataLoader pour l’entraînement
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Nombre de features et nombre de classes
num_features = X_train_tensor.shape[2]
num_classes = y_train_tensor.unique().numel()

# Instanciation du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_features=num_features, num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Suivi des courbes de loss
train_losses = []
val_losses = []

for epoch in range(num_epochs):
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
    
    # Calcul de la loss sur le jeu de validation
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_tensor.to(device))
        loss_val = criterion(outputs_val, y_val_tensor.to(device)).item()
    val_losses.append(loss_val)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_train_loss:.4f} - Validation Loss: {loss_val:.4f}")

# Affichage des courbes de loss
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Époch')
plt.ylabel('Loss')
plt.title("Évolution de la loss")
plt.legend()
plt.tight_layout()
plt.show()

# Évaluation sur le jeu de validation
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor.to(device))
    _, y_pred_tensor = torch.max(outputs, 1)
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
