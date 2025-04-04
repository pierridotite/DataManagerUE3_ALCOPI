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
original_counts = data.iloc[:, -1].value_counts()  # Sauvegarde des données de base

# Augmentation pour équilibrer les espèces
species_counts = data.iloc[:, -1].value_counts()
max_count = species_counts.max()
augmented_frames = []
for specie in species_counts.index:
    df_specie = data[data.iloc[:, -1] == specie]
    df_oversampled = df_specie.sample(max_count, replace=True, random_state=42)
    augmented_frames.append(df_oversampled)
data = pd.concat(augmented_frames, axis=0).reset_index(drop=True)
# Fin de l'augmentation

# Visualisation de l'augmentation
augmented_counts = data.iloc[:, -1].value_counts()
original_counts = original_counts.reindex(augmented_counts.index)  # aligner les indices
augmentation_numbers = augmented_counts - original_counts

plt.figure(figsize=(8,6))
species = augmented_counts.index
plt.bar(species, original_counts, color='blue', label='Data de base')
plt.bar(species, augmentation_numbers, bottom=original_counts, color='orange', label='Augmentation')
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Visualisation de l'augmentation")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# On considère que la dernière colonne est la cible
X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
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
plt.xticks(range(0, len(importances), step), X.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
