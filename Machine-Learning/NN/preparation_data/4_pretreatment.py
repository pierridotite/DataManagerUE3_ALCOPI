import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from scipy.signal import savgol_filter
from sklearn.pipeline import make_pipeline

# Pour la reproductibilité
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# 1. Chargement et préparation des données
# =============================================================================

data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

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

# Séparation en features et target (validation simple)
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]
X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# Encodage de la cible si nécessaire (pour assurer des classes numériques)
if y_train.dtype == object:
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train))
    y_val = pd.Series(le.transform(y_val))

# =============================================================================
# 2. Affichage de l'histogramme des effectifs par espèce
# =============================================================================

# Comptage des effectifs dans le jeu complet, training et validation
full_counts = data[target_col].value_counts().sort_index()
train_counts = training_data[target_col].value_counts().sort_index()
val_counts   = validation_data[target_col].value_counts().sort_index()

# On trie les classes selon les effectifs décroissants dans le training set
order = train_counts.sort_values(ascending=False).index
train_counts = train_counts.reindex(order)
val_counts   = val_counts.reindex(order, fill_value=0)

species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
plt.bar(x, val_counts, width, color='green', label='Validation dataset (6 per species)')
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training dataset')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme : Effectifs par espèce (Training + Validation)")
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# 3. Définition des fonctions de prétraitement
# =============================================================================

def identity(X):
    """Retourne X sans modification."""
    return X.copy()

def snv(X):
    """Application de la transformation SNV (Standard Normal Variate) sur chaque ligne."""
    X_array = X.values
    X_snv_array = np.array([ (row - row.mean())/row.std() if row.std() != 0 else row - row.mean() for row in X_array])
    return pd.DataFrame(X_snv_array, index=X.index, columns=X.columns)

def first_derivative(X):
    """Calcul de la 1ère dérivée pour chaque échantillon."""
    X_array = X.values
    X_deriv = np.gradient(X_array, axis=1)
    return pd.DataFrame(X_deriv, index=X.index, columns=X.columns)

def second_derivative(X):
    """Calcul de la 2ème dérivée pour chaque échantillon."""
    X_array = X.values
    first_deriv = np.gradient(X_array, axis=1)
    second_deriv = np.gradient(first_deriv, axis=1)
    return pd.DataFrame(second_deriv, index=X.index, columns=X.columns)

def snv_first_derivative(X):
    """SNV suivi de la 1ère dérivée."""
    return first_derivative(snv(X))

def snv_second_derivative(X):
    """SNV suivi de la 2ème dérivée."""
    return second_derivative(snv(X))

def savitzky_golay(X, window_length=7, polyorder=2):
    """Filtrage Savitzky-Golay pour le lissage des données."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), 1, X.values)
    return pd.DataFrame(X_sg, index=X.index, columns=X.columns)

def savgol_first_derivative(X, window_length=7, polyorder=2):
    """Filtrage Savitzky-Golay suivi du calcul de la 1ère dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg1 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=1), 1, X.values)
    return pd.DataFrame(X_sg1, index=X.index, columns=X.columns)

def savgol_second_derivative(X, window_length=7, polyorder=2):
    """Filtrage Savitzky-Golay suivi du calcul de la 2ème dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg2 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=2), 1, X.values)
    return pd.DataFrame(X_sg2, index=X.index, columns=X.columns)

def standard_scaling_train_val(X_train, X_val):
    """Standardisation (column-wise) avec ajustement sur le training set."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    return X_train_scaled, X_val_scaled

# =============================================================================
# 4. Dictionnaire des méthodes de prétraitement à tester
# =============================================================================

preprocessing_methods = {
    "Aucun": identity,
    "SNV": snv,
    "1ère dérivée": first_derivative,
    "2ème dérivée": second_derivative,
    "SNV + 1ère dérivée": snv_first_derivative,
    "SNV + 2ème dérivée": snv_second_derivative,
    "Savitzky-Golay": savitzky_golay,
    "Savitzky-Golay + 1ère dérivée": savgol_first_derivative,
    "Savitzky-Golay + 2ème dérivée": savgol_second_derivative,
    "Standard Scaling": None  # Traitement particulier
}

# =============================================================================
# 5. Définition du modèle CNN de base pour les données tabulaires
# =============================================================================
# Le CNN ci-dessous prend en entrée des vecteurs de features transformés en tenseurs
# de forme (n_samples, 1, n_features)

class CNN(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.5):
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
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Paramètres d'entraînement pour le CNN
hp = {
    "lr": 0.005,             # mise à jour : learning rate
    "batch_size": 64,        # mise à jour : taille de lot
    "num_epochs": 100,       # inchangé
    "dropout_rate": 0.0,     # mise à jour : pas de dropout
    "weight_decay": 0.01     # mise à jour : régularisation L2
}
batch_size = hp["batch_size"]
num_epochs = hp["num_epochs"]

# =============================================================================
# 6. Test de chaque prétraitement avec validation simple (CNN)
# =============================================================================

cnn_results = {}  # Stocke les accuracies pour chaque méthode

print("\n====== Validation simple avec CNN ======")
for method_name, func in preprocessing_methods.items():
    print(f"\n--- Prétraitement: {method_name} ---")
    # Gestion particulière pour Standard Scaling
    if method_name == "Standard Scaling":
        X_train_trans, X_val_trans = standard_scaling_train_val(X_train, X_val)
    else:
        X_train_trans = func(X_train)
        X_val_trans = func(X_val)
        # New: Apply standard scaling after transformation
        X_train_trans, X_val_trans = standard_scaling_train_val(X_train_trans, X_val_trans)
    
    # Conversion en arrays numpy
    X_train_np = X_train_trans.values
    X_val_np = X_val_trans.values
    y_train_np = y_train.values
    y_val_np = y_val.values
    
    # Conversion en tenseurs et ajout de la dimension canal pour le CNN
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.long)
    
    # Création du DataLoader pour l'entraînement
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    num_features = X_train_tensor.shape[2]
    num_classes = torch.unique(y_train_tensor).numel()
    
    # Instanciation du modèle CNN avec mise à jour du dropout_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_features=num_features, num_classes=num_classes, dropout_rate=hp["dropout_rate"])
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    
    # Entraînement sur num_epochs époques avec suivi validation
    model.train()
    train_losses = []      # Stocke la loss d'entraînement par époque
    val_losses = []        # Stocke la loss de validation par époque
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        # Évaluation sur le jeu de validation pour cette époque
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_loss = criterion(val_outputs, y_val_tensor.to(device)).item()
            val_losses.append(val_loss)
        model.train()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Training Loss={epoch_loss:.4f}, Validation Loss={val_loss:.4f}")
    # Nouveau : affichage des courbes d'entraînement et validation
    epochs = np.arange(1, num_epochs+1)
    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Courbes d'entraînement et validation ({method_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Évaluation sur le jeu de validation
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_tensor.to(device))
        _, y_pred_tensor = torch.max(outputs_val, 1)
        y_pred = y_pred_tensor.cpu().numpy()
    acc = accuracy_score(y_val_np, y_pred)
    cnn_results[method_name] = {"accuracy": acc}
    
    print("Classification Report:")
    print(classification_report(y_val_np, y_pred))
    cm = confusion_matrix(y_val_np, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion ({method_name})")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables")
    plt.tight_layout()
    plt.show()
    
    print(f"Score de validation simple ({method_name}) : {acc:.4f}")

# Affichage des résultats sous forme de DataFrame et graphique
cnn_results_df = pd.DataFrame(cnn_results).T
print("\nComparaison des accuracies (Validation simple) :")
print(cnn_results_df)

plt.figure(figsize=(8,4))
plt.bar(cnn_results_df.index, cnn_results_df['accuracy'], color='skyblue')
plt.ylabel("Accuracy")
plt.title("Comparaison des accuracies selon le prétraitement (CNN)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

