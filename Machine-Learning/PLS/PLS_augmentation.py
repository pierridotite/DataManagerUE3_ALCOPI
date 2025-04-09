# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from chemotools.derivative import SavitzkyGolay
from chemotools.scatter import StandardNormalVariate
from chemotools.feature_selection import RangeCut
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

### CHARGEMENT DES DONNEES ###

# Load the combined data
data = pd.read_csv('/Users/constance/Documents/GitHub/DataManagerUE3_ALCOPI/Data/combined_data.csv', 
                      sep=',', 
                      index_col=0)
print(data)

### SEPARATION DES VARIABLES EXPLICATIVES (SPECTRE) ET CIBLE (CLASS) ###

# Séparation du dataset AVANT toute transformation
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Concaténer pour traiter uniquement l'entraînement
train = pd.concat([X_train, y_train], axis=1)

original_counts = train.iloc[:, -1].value_counts()  # données de base d'entraînement
data_orig = train.copy()  # bornes pour l'augmentation

### NORMALISATION DES DONNEES ###

# Standardisation des variables explicatives AVANT le split
scaler = StandardScaler()
X_scaled_total = scaler.fit_transform(X)

# On garde les noms de colonnes pendant la normalisation
X_scaled_total = pd.DataFrame(X_scaled_total, columns=X.columns)

# Vérifier la moyenne et l'écart-type après standardisation
print("Moyenne après standardisation :", X_scaled_total.mean(axis=0).round(2))
print("Écart-type après standardisation :", X_scaled_total.std(axis=0).round(2))

### SPLIT AVANT AUGMENTATION ###

# Séparation à nouveau après normalisation
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled_total, y, test_size=0.3, random_state=42, stratify=y)

# Concatenation pour traiter uniquement l'entraînement
train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
train_scaled['class'] = y_train.values

# Visualisation avant et après l'augmentation (optionnel)
original_counts = train_scaled['class'].value_counts()  # données de base d'entraînement
max_count = original_counts.max()

### AUGMENTATION ###

# Oversampling
species_counts = train_scaled['class'].value_counts()
max_count = species_counts.max()
augmented_frames = []
for specie in species_counts.index:
    df_specie = train_scaled[train_scaled['class'] == specie]
    df_oversampled = df_specie.sample(max_count, replace=True, random_state=42)
    augmented_frames.append(df_oversampled)
train_scaled_aug = pd.concat(augmented_frames, axis=0).reset_index(drop=True)

# Augmentation aléatoire
n_random = int(round((2/3) * max_count))
random_aug_frames = []
features = data_orig.columns[:-1]

for specie in original_counts.index:
    df_specie_orig = data_orig[data_orig['species'] == specie]
    for _ in range(n_random):
        new_row = {}
        for col in features:
            lower = df_specie_orig[col].min()
            upper = df_specie_orig[col].max()
            new_row[col] = np.random.uniform(lower, upper)
        new_row['species'] = specie
        random_aug_frames.append(new_row)

random_aug = pd.DataFrame(random_aug_frames)
train_scaled_aug = pd.concat([train_scaled_aug, random_aug], axis=0).reset_index(drop=True)

# Visualisation après augmentation
augmented_counts = train_scaled_aug['species'].value_counts()
oversample_numbers = max_count - original_counts
random_aug_counts = pd.Series(n_random, index=original_counts.index)

plt.figure(figsize=(8,6))
species = augmented_counts.index
plt.bar(species, original_counts, color='blue', label='Données de base')
plt.bar(species, oversample_numbers, bottom=original_counts, color='orange', label='Oversampling')
plt.bar(species, random_aug_counts, bottom=original_counts + oversample_numbers, color='green', label='Augmentation Aléatoire')
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Visualisation de l'augmentation")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

### MODELISATION ###

# Re-séparer les X et y à partir des données augmentées
X_train_aug = train_scaled.iloc[:, :-1]
y_train_aug = train_scaled.iloc[:, -1]

# Transformation des classes cibles en nombres (encodage)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train_aug)

# Tester plusieurs nombres de composantes
n_components_list = range(1, 21)
scores = []

for n in n_components_list:
    pls = PLSRegression(n_components=n)
    score = cross_val_score(pls, X_train_aug, y_encoded, cv=5, scoring='r2').mean()
    scores.append(score)

# Trouver le nombre optimal de composantes
best_n_components = n_components_list[np.argmax(scores)]
print(f"Nombre optimal de composantes : {best_n_components}")

# Entraînement du modèle avec le meilleur nombre de composantes
pls_final = PLSRegression(n_components=best_n_components)
pls_final.fit(X_train_aug, y_encoded)

# Prédiction sur le test
y_pred = pls_final.predict(X_test_scaled)

# Évaluer la performance
print("Score R2 :", pls_final.score(X_test_scaled, label_encoder.transform(y_test)))

from sklearn.model_selection import GridSearchCV

# Paramètres à tester
param_grid = {'n_components': range(1, 21)}  # Nombres de composantes à tester

pls_da = PLSRegression()
grid_search = GridSearchCV(pls_da, param_grid, cv=5, scoring='accuracy')  # Utilisation de la précision comme score
grid_search.fit(X_train_aug, pd.get_dummies(y_train_aug))

print(f"Meilleur nombre de composantes : {grid_search.best_params_['n_components']}")
