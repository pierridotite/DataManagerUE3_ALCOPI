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

# La dernière colonne correspond à la cible
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

# Séparation en features et target pour l'entraînement
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

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

# Définition de la grille de paramètres pour l'augmentation
param_grid_aug = {
    'scale_min': [0.95, 0.9],
    'scale_max': [1.05, 1.1],
    'noise_std': [0.02, 0.05],
    'shift_min': [-2, -3],
    'shift_max': [2, 3]
}

# Nombre fixe d'exemples augmentés par classe
n_aug_per_class = 10

# Liste pour stocker les résultats
results = []

# Boucle sur chaque combinaison de paramètres
for params in ParameterGrid(param_grid_aug):
    # Génération des données augmentées pour chaque classe
    augmented_X_list = []
    augmented_y_list = []
    
    for cls in y_train.unique():
        # Récupérer les indices des exemples de la classe
        cls_indices = X_train[y_train == cls].index
        # Échantillonnage avec remise pour obtenir exactement n_aug_per_class exemples
        chosen_indices = np.random.choice(cls_indices, size=n_aug_per_class, replace=True)
        for idx in chosen_indices:
            spectrum = X_train.loc[idx].values.astype(float)
            aug_spec = augment_spectrum(spectrum, params['scale_min'], params['scale_max'],
                                        params['noise_std'], params['shift_min'], params['shift_max'])
            augmented_X_list.append(aug_spec)
            augmented_y_list.append(cls)
    
    # Reconstruction des DataFrames pour les exemples augmentés
    X_train_aug = pd.DataFrame(augmented_X_list, columns=X_train.columns)
    y_train_aug = pd.Series(augmented_y_list, name=target_col)
    
    # Combinaison des données originales et augmentées
    X_train_final = pd.concat([X_train, X_train_aug], ignore_index=True)
    y_train_final = pd.concat([y_train, y_train_aug], ignore_index=True)
    
    # Entraînement du modèle avec ces données
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_final, y_train_final)
    
    # Préparation du jeu de validation
    X_val = validation_data.drop(columns=[target_col])
    y_val = validation_data[target_col]
    
    # Prédiction et évaluation sur le jeu de validation
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    # Stocker le résultat pour cette configuration
    results.append({
        'params': params,
        'accuracy': acc
    })
    print("Params:", params, "-> Accuracy:", acc)

# Transformation des résultats en DataFrame et affichage des meilleurs paramètres
results_df = pd.DataFrame(results)
print("\nRésultats de la grid search (triés par accuracy décroissante) :")
print(results_df.sort_values(by='accuracy', ascending=False))
