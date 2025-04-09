import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import savgol_filter

# ================================
# 1. Chargement et préparation des données
# ================================

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

# Séparation en features et target
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# ================================
# 2. Affichage de l'histogramme des effectifs par espèce
# ================================

full_counts = data[target_col].value_counts().sort_index()
train_counts = training_data[target_col].value_counts().sort_index()
val_counts = validation_data[target_col].value_counts().sort_index()

# On trie les espèces selon les effectifs décroissants dans le training dataset
order = train_counts.sort_values(ascending=False).index
train_counts = train_counts.reindex(order)
val_counts = val_counts.reindex(order, fill_value=0)

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

# ================================
# 3. Entraînement et évaluation du modèle de base (sans prétraitement supplémentaire)
# ================================

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

print("=== Modèle sans prétraitement supplémentaire ===")
print("Classification Report:\n", classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Aucun prétraitement)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables (Aucun prétraitement)")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ================================
# 4. Définition des fonctions de prétraitement
# ================================

def identity(X):
    """Aucun prétraitement."""
    return X.copy()

def snv(X):
    """Transformation SNV (Standard Normal Variate) ligne par ligne."""
    X_array = X.values
    X_snv_array = np.array([(row - row.mean())/row.std() if row.std() != 0 else row - row.mean() for row in X_array])
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
    """Savitzky-Golay suivi du calcul de la 1ère dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg1 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=1), 1, X.values)
    return pd.DataFrame(X_sg1, index=X.index, columns=X.columns)

def savgol_second_derivative(X, window_length=7, polyorder=2):
    """Savitzky-Golay suivi du calcul de la 2ème dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg2 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=2), 1, X.values)
    return pd.DataFrame(X_sg2, index=X.index, columns=X.columns)

def standard_scaling_train_val(X_train, X_val):
    """Standardisation column-wise (calculée sur X_train et appliquée sur X_val)."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    return X_train_scaled, X_val_scaled

# ================================
# 5. Dictionnaire des méthodes de prétraitement à tester
# ================================

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

# ================================
# 6. Création d'un dataset étendu en concaténant les prétraitements au dataset de base
# ================================

# On démarre avec les données de base
X_train_extended = X_train.copy()
X_val_extended = X_val.copy()

# On applique chaque transformation et on concatène les résultats
for method_name, func in preprocessing_methods.items():
    # Pour "Aucun", on ne fait rien (les données de base sont déjà présentes)
    if method_name == "Aucun":
        continue
    if method_name == "Standard Scaling":
        X_train_trans, X_val_trans = standard_scaling_train_val(X_train, X_val)
    else:
        X_train_trans = func(X_train)
        X_val_trans = func(X_val)
    
    # Renommer les colonnes pour identifier la transformation
    X_train_trans = X_train_trans.add_prefix(f"{method_name}_")
    X_val_trans = X_val_trans.add_prefix(f"{method_name}_")
    
    # Concaténer avec le dataset de base
    X_train_extended = pd.concat([X_train_extended, X_train_trans], axis=1)
    X_val_extended = pd.concat([X_val_extended, X_val_trans], axis=1)

print("Shape de X_train étendu :", X_train_extended.shape)
print("Shape de X_val étendu :", X_val_extended.shape)

# ================================
# 7. Entraînement et évaluation du modèle sur le dataset étendu
# ================================

rf_extended = RandomForestClassifier(random_state=42)
rf_extended.fit(X_train_extended, y_train)
y_pred_ext = rf_extended.predict(X_val_extended)

print("=== Modèle sur dataset étendu ===")
print("Classification Report:\n", classification_report(y_val, y_pred_ext))

cm_ext = confusion_matrix(y_val, y_pred_ext)
plt.figure(figsize=(8,6))
sns.heatmap(cm_ext, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Dataset étendu)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Ajout de l'affichage de l'importance des variables pour le dataset étendu
importances_ext = rf_extended.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables (Dataset étendu)")
plt.bar(range(len(importances_ext)), importances_ext, align="center")
step = max(1, len(importances_ext) // 10)
plt.xticks(range(0, len(importances_ext), step), X_train_extended.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
