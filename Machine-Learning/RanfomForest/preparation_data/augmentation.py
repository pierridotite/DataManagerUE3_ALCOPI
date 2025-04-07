import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import math

# Charger les données
data = pd.read_csv(r'c:\Users\ACER\OneDrive\Bureau\DataManagerUE3_ALCOPI\Data\combined_data.csv')
# Séparation du dataset AVANT toute transformation
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Concaténer pour traiter uniquement l'entraînement
train = pd.concat([X_train, y_train], axis=1)

original_counts = train.iloc[:, -1].value_counts()  # données de base d'entraînement
data_orig = train.copy()  # bornes pour l'augmentation

# Augmentation pour équilibrer les espèces sur l'ensemble d'entraînement
species_counts = train.iloc[:, -1].value_counts()
max_count = species_counts.max()
augmented_frames = []
for specie in species_counts.index:
    df_specie = train[train.iloc[:, -1] == specie]
    df_oversampled = df_specie.sample(max_count, replace=True, random_state=42)
    augmented_frames.append(df_oversampled)
train = pd.concat(augmented_frames, axis=0).reset_index(drop=True)
# Fin de l'oversampling existant

# --- Nouvelle étape d'augmentation aléatoire ---
# Calcul du nombre de valeurs aléatoires par espèce : 
# On souhaite que random_count/(max_count + random_count)=0.4, donc random_count = (2/3)*max_count
n_random = int(round((2/3) * max_count))
random_aug_frames = []
features = data_orig.columns[:-1]
for specie in original_counts.index:
    df_specie_orig = data_orig[data_orig.iloc[:, -1] == specie]
    for _ in range(n_random):
        new_row = {}
        for col in features:
            lower = df_specie_orig[col].min()
            upper = df_specie_orig[col].max()
            new_row[col] = np.random.uniform(lower, upper)
        new_row[train.columns[-1]] = specie  # affecter l'étiquette
        random_aug_frames.append(new_row)
random_aug = pd.DataFrame(random_aug_frames)
train = pd.concat([train, random_aug], axis=0).reset_index(drop=True)
# --- Fin de l'augmentation aléatoire ---

# Visualisation de l'augmentation
augmented_counts = train.iloc[:, -1].value_counts()
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

# On considère que la dernière colonne est la cible pour l'entraînement
X_train_aug = train.iloc[:, :-1]
y_train_aug = train.iloc[:, -1]
# Le test reste inchangé (X_test, y_test)

# Modèle Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_aug, y_train_aug)
y_pred = rf.predict(X_test)

# Affichage des résultats dans la console
accuracy = accuracy_score(y_test, y_pred)
print("Précision :", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)
print(classification_report(y_test, y_pred))

# Plot de la matrice de confusion
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.show()

# Validation supplémentaire pour data analyst : calcul de R carré
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test_enc = le.fit_transform(y_test)
y_pred_enc = le.transform(y_pred)
r2 = r2_score(y_test_enc, y_pred_enc)
print("R carré :", r2)

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
# Limiter le nombre de labels affichés sur l'axe x
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train_aug.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
