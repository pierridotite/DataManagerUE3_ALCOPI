import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid

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
      - une mise à l'échelle aléatoire
      - l'ajout de bruit gaussien
      - un décalage spectral
    """
    # Mise à l'échelle aléatoire
    scale_factor = np.random.uniform(scale_min, scale_max)
    spectrum_aug = spectrum * scale_factor

    # Ajout de bruit gaussien
    noise = np.random.normal(0, noise_std, size=spectrum.shape)
    spectrum_aug += noise

    # Décalage spectral
    shift = np.random.randint(shift_min, shift_max + 1)
    spectrum_aug = np.roll(spectrum_aug, shift)
    
    return spectrum_aug

# ================================
# 3. Recherche en grille sur les paramètres d'augmentation
# ================================
# Grille de paramètres plus complète
param_grid_aug = {
    'scale_min': [0.80,0.85],
    'scale_max': [1.15,1.20],
    'noise_std': [0.005, 0.01, 0.015],
    'shift_min': [-8, -5, -3],
    'shift_max': [3, 5, 8]
}

# Nombre fixe d'exemples augmentés par classe
n_aug_per_class = 10

results = []

print("Début de la grid search sur {} combinaisons...".format(len(list(ParameterGrid(param_grid_aug)))))

for params in ParameterGrid(param_grid_aug):
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
    
    # Entraînement du modèle avec cette configuration d'augmentation
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_final, y_train_final)
    
    # Évaluation sur le jeu de validation
    y_pred = clf.predict(X_val)
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
# 4. Visualisation d'un exemple d'augmentation
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
# 5. Entraînement final avec la meilleure configuration d'augmentation
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

# Entraînement final du Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_final, y_train_final)

y_pred = rf.predict(X_val)

print("\nClassification Report avec la meilleure configuration d'augmentation :")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Calcul du R² (après encodage des labels)
le = LabelEncoder()
y_val_enc = le.fit_transform(y_val)
y_pred_enc = le.transform(y_pred)
r2 = r2_score(y_val_enc, y_pred_enc)
print("R² :", r2)

# Affichage des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
