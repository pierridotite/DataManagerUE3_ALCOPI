import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns  # Ajouté

# Définition des fonctions de prétraitement
def identity(X):
    return X.copy()

def snv(X):
    X_array = X.values
    X_snv_array = np.array([ (row - row.mean())/row.std() if row.std() != 0 else row - row.mean() for row in X_array])
    return pd.DataFrame(X_snv_array, index=X.index, columns=X.columns)

def first_derivative(X):
    X_array = X.values
    X_deriv = np.gradient(X_array, axis=1)
    return pd.DataFrame(X_deriv, index=X.index, columns=X.columns)

def second_derivative(X):
    X_array = X.values
    first_deriv = np.gradient(X_array, axis=1)
    second_deriv = np.gradient(first_deriv, axis=1)
    return pd.DataFrame(second_deriv, index=X.index, columns=X.columns)

def savitzky_golay(X, window_length=7, polyorder=2):
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), 1, X.values)
    return pd.DataFrame(X_sg, index=X.index, columns=X.columns)

def standard_scaling(X):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

# Définition du dictionnaire des prétraitements (hors "Aucun")
preprocessing_methods = {
    "SNV": snv,
    "1ère dérivée": first_derivative,
    "2ème dérivée": second_derivative,
    "Savitzky-Golay": savitzky_golay
}

# Chargement et préparation des données
data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)
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
training_data = data.drop(validation_data.index)

X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]
X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

if y_train.dtype == object:
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train))
    y_val = pd.Series(le.transform(y_val))

# Création du dataset étendu par concaténation des prétraitements
X_train_extended = X_train.copy()
X_val_extended = X_val.copy()
for method_name, func in preprocessing_methods.items():
    X_train_proc = func(X_train)
    X_val_proc = func(X_val)
    # Appliquer standard scaling sur les données transformées
    X_train_proc = standard_scaling(X_train_proc)
    X_val_proc = standard_scaling(X_val_proc)
    X_train_proc = X_train_proc.add_prefix(f"{method_name}_")
    X_val_proc = X_val_proc.add_prefix(f"{method_name}_")
    X_train_extended = pd.concat([X_train_extended, X_train_proc], axis=1)
    X_val_extended = pd.concat([X_val_extended, X_val_proc], axis=1)

# Standard scaling sur l'ensemble étendu
X_train_extended = standard_scaling(X_train_extended)
X_val_extended = standard_scaling(X_val_extended)

print("Shape de X_train étendu :", X_train_extended.shape)
print("Shape de X_val étendu :", X_val_extended.shape)

# Préparation pour le CNN : conversion des DataFrames en tenseurs et ajout de la dimension canal
X_train_np = X_train_extended.values
X_val_np = X_val_extended.values
y_train_np = y_train.values
y_val_np = y_val.values

X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.long)

# Définition du modèle CNN de base
class CNN(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

batch_size = 32
num_epochs = 500
learning_rate = 0.001

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_features = X_train_tensor.shape[2]
num_classes = len(np.unique(y_train_np))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_features=num_features, num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement avec suivi de la loss de validation à chaque époque
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
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation après chaque époque
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_tensor.to(device))
        loss_val = criterion(outputs_val, y_val_tensor.to(device)).item()
        val_losses.append(loss_val)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f} - Validation Loss: {loss_val:.4f}")

model.eval()
with torch.no_grad():
    outputs_val = model(X_val_tensor.to(device))
    _, y_pred_tensor = torch.max(outputs_val, 1)
    y_pred = y_pred_tensor.cpu().numpy()
acc = accuracy_score(y_val_np, y_pred)
print("Accuracy sur validation (dataset étendu) :", acc)

# Affichage du graphique de la courbe d'entraînement et de validation
epochs_arr = np.arange(1, num_epochs+1)
plt.figure()
plt.plot(epochs_arr, train_losses, marker='o', linestyle='-', color='purple', label="Training Loss")
plt.plot(epochs_arr, val_losses, marker='s', linestyle='--', color='red', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Courbes d'entraînement et validation - Dataset étendu")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcul et affichage de la matrice de confusion
cm = confusion_matrix(y_val_np, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion - Dataset étendu")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()