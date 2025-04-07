import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Charger les données
data_src = r'Data\combined_data.csv'
data_full = pd.read_csv(data_src)
# Séparation du data set dès le début
X_full = data_full.iloc[:, :-1]
y_full = data_full.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42)
# Concaténer pour traiter uniquement l’entraînement
train = pd.concat([X_train, y_train], axis=1)

# Création du data set de validation équilibré (5 échantillons par espèce)
val_data = pd.concat([X_test, y_test], axis=1)
balanced_val_frames = []
for specie in val_data.iloc[:, -1].unique():
    df_specie = val_data[val_data.iloc[:, -1] == specie]
    df_val_balanced = df_specie.sample(n=5, replace=True, random_state=42)
    balanced_val_frames.append(df_val_balanced)
val_balanced = pd.concat(balanced_val_frames, axis=0).reset_index(drop=True)
X_val_bal = val_balanced.iloc[:, :-1]
y_val_bal = val_balanced.iloc[:, -1]

# Sauvegarde des données de base d'entraînement
base_counts = train.iloc[:, -1].value_counts()

# Augmentation pour équilibrer les espèces sur l'ensemble d’entraînement
species_counts = base_counts.copy()
max_count = species_counts.max()  # nombre cible pour chaque espèce
augmented_frames = []
for specie in species_counts.index:
    df_specie = train[train.iloc[:, -1] == specie]
    # oversampling pour atteindre max_count
    df_oversampled = df_specie.sample(max_count, replace=True, random_state=42)
    augmented_frames.append(df_oversampled)
train_balanced = pd.concat(augmented_frames, axis=0).reset_index(drop=True)
# Calcul des nombres d'augmentation
balanced_counts = train_balanced.iloc[:, -1].value_counts()  # seront identiques pour chaque espèce
augmentation_numbers = balanced_counts - base_counts

# Visualisation sur une seule barre par espèce (empilée)
import numpy as np
species = balanced_counts.index
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
plt.bar(x, base_counts[species], width, color='blue', label='Data de base')
plt.bar(x, augmentation_numbers[species], width, bottom=base_counts[species], color='orange', label='Augmentation')

# Annotation du nombre d'exemples dans le data set de validation équilibré
val_counts = y_val_bal.value_counts().reindex(species)  # tous devraient être 5
for i, specie in enumerate(species):
    plt.text(x[i],
             base_counts[specie] + augmentation_numbers[specie] + 2,
             f'Validation: {val_counts[specie]}',
             ha='center', va='bottom', color='red', fontsize=10)

plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Répartition équilibrée par espèce (Entraînement) avec validation")
plt.xticks(x, species, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# On considère que la dernière colonne est la cible
X_train_bal = train_balanced.iloc[:, :-1]
y_train_bal = train_balanced.iloc[:, -1]

# Modèle Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_bal, y_train_bal)
y_pred = rf.predict(X_val_bal)  # utilisation de la validation équilibrée

# Affichage des résultats dans la console
accuracy = accuracy_score(y_val_bal, y_pred)
print("Précision :", accuracy)
cm = confusion_matrix(y_val_bal, y_pred)
print("Matrice de confusion :\n", cm)
print(classification_report(y_val_bal, y_pred))

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
y_val_enc = le.fit_transform(y_val_bal)
y_pred_enc = le.transform(y_pred)
r2 = r2_score(y_val_enc, y_pred_enc)
print("R carré :", r2)

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
# Limiter le nombre de labels affichés sur l'axe x
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train_bal.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
