import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

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

# Visualisation combinée du train (stacké) et du data set de validation dans un seul plot

# Calcul des comptes pour le train
base = original_counts
oversample_numbers = max_count - original_counts
random_aug_counts = pd.Series(n_random, index=original_counts.index)
train_totals = base + oversample_numbers + random_aug_counts

# Comptes pour le data set de validation
validation_counts = y_test.value_counts().reindex(original_counts.index)

import numpy as np
species = original_counts.index
x = np.arange(len(species))
width = 0.35

plt.figure(figsize=(10,6))
# Barres empilées pour le train (décalées à gauche)
plt.bar(x - width/2, base, width, color='blue', label='Train: Données de base')
plt.bar(x - width/2, oversample_numbers, width, bottom=base, color='orange', label='Train: Oversampling')
plt.bar(x - width/2, random_aug_counts, width, bottom=base+oversample_numbers, color='green', label='Train: Augmentation Aléatoire')
# Barre simple pour le data set de validation (décalée à droite)
plt.bar(x + width/2, validation_counts, width, color='red', label='Validation')

plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Répartition par espèce: Train vs Validation")
plt.xticks(x, species, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Use base training + noise augmented rows for training
balanced_train = pd.concat([train_data, noise_augmented]).reset_index(drop=True)
X_train = balanced_train.drop(columns=[label_col])
y_train = balanced_train[label_col]
X_test = test_data.drop(columns=[label_col])
y_test = test_data[label_col]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForestClassifier training and evaluation
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
plt.tight_layout()
plt.show()

# Compute R² score after encoding species labels numerically
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test_num = le.fit_transform(y_test)
y_pred_num = le.transform(y_pred)
r2 = r2_score(y_test_num, y_pred_num)
print("R² score:", r2)
