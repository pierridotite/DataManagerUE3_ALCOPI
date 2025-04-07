import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder

# ================================
# 1. Chargement et préparation des données
# ================================
data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

# Création du jeu de validation : 5 individus par espèce
validation_frames = []
for specie in data[target_col].unique():
    specie_df = data[data[target_col] == specie]
    if len(specie_df) >= 5:
        val_specie = specie_df.sample(n=5, random_state=42)
    else:
        val_specie = specie_df.sample(n=5, replace=True, random_state=42)
    validation_frames.append(val_specie)
validation_data = pd.concat(validation_frames)

# Le jeu d'entraînement correspond aux données restantes
training_data = data.drop(validation_data.index)

# Séparation en features et target
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

# ================================
# 2. Augmentation des données spectrales (nombre fixe par classe)
# ================================

def augment_spectrum(spectrum, scale_range=(0.90, 1.10), noise_std=0.02, shift_range=(-2, 2)):
    """
    Augmente un spectre par mise à l'échelle, ajout de bruit et décalage spectral.
    
    Parameters:
        spectrum (array-like): vecteur de données spectrales.
        scale_range (tuple): intervalle pour le facteur de mise à l'échelle.
        noise_std (float): écart-type du bruit gaussien.
        shift_range (tuple): intervalle (min, max) pour le décalage spectral (en indices).
        
    Returns:
        numpy.ndarray: spectre augmenté.
    """
    # 1. Mise à l'échelle aléatoire
    scale_factor = np.random.uniform(*scale_range)
    spectrum_aug = spectrum * scale_factor
    
    # 2. Ajout de bruit gaussien
    noise = np.random.normal(0, noise_std, size=spectrum.shape)
    spectrum_aug += noise
    
    # 3. Décalage spectral
    shift = np.random.randint(shift_range[0], shift_range[1] + 1)
    spectrum_aug = np.roll(spectrum_aug, shift)
    
    return spectrum_aug

# Nombre fixe d'augmentations par classe
n_aug_per_class = 10

augmented_X_list = []
augmented_y_list = []

# Boucle sur chaque classe pour générer un nombre fixe d'exemples augmentés
for cls in y_train.unique():
    # Indices correspondant à la classe considérée
    cls_indices = X_train[y_train == cls].index
    # Echantillonnage avec remise pour obtenir exactement n_aug_per_class exemples
    chosen_indices = np.random.choice(cls_indices, size=n_aug_per_class, replace=True)
    for idx in chosen_indices:
        spectrum = X_train.loc[idx].values.astype(float)
        augmented_spectrum = augment_spectrum(spectrum)
        augmented_X_list.append(augmented_spectrum)
        augmented_y_list.append(cls)

# Reconstruction des DataFrames pour les données augmentées
X_train_aug = pd.DataFrame(augmented_X_list, columns=X_train.columns)
y_train_aug = pd.Series(augmented_y_list, name=target_col)

# Combinaison des données originales et augmentées
X_train_final = pd.concat([X_train, X_train_aug], ignore_index=True)
y_train_final = pd.concat([y_train, y_train_aug], ignore_index=True)

# ================================
# 3. Visualisation des effectifs par espèce (optionnelle)
# ================================
full_counts = data[target_col].value_counts().sort_index()
train_counts = training_data[target_col].value_counts().sort_index()
val_counts = validation_data[target_col].value_counts().sort_index()

order = train_counts.sort_values(ascending=False).index
train_counts = train_counts.reindex(order)
val_counts = val_counts.reindex(order, fill_value=0)

# Nombre d'exemples augmentés par classe (fixe : n_aug_per_class)
aug_counts = pd.Series({cls: n_aug_per_class for cls in order})

species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
plt.bar(x, val_counts, width, color='green', label='Validation (5 par espèce)')
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training')
plt.bar(x, aug_counts, width, bottom=val_counts.values + train_counts.values, 
        color='orange', label='Augmentation')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Effectifs par espèce (Training + Validation + Augmentation)")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 4. Entraînement et évaluation du modèle
# ================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_final, y_train_final)

# Préparation du jeu de validation
X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

y_pred = rf.predict(X_val)

print("Classification Report:\n", classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Calcul du R² après encodage des labels
le = LabelEncoder()
y_val_enc = le.fit_transform(y_val)
y_pred_enc = le.transform(y_pred)
r2 = r2_score(y_val_enc, y_pred_enc)
print("R² :", r2)

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
