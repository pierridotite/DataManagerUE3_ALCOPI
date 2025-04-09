import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler  # Importer RandomOverSampler
from imblearn.pipeline import Pipeline  # Pour intégrer l'oversampling dans la validation croisée
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

# ================================
# 1. Chargement et préparation des données
# ================================

# Charger les données
data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

# Création du jeu de validation : 6 individus par espèce
validation_frames = []
for specie in data[target_col].unique():
    specie_df = data[data[target_col] == specie]
    # Si possible, on prélève sans remise ; sinon, avec remise
    if len(specie_df) >= 6:
        val_specie = specie_df.sample(n=6, random_state=42)
    else:
        val_specie = specie_df.sample(n=6, replace=True, random_state=42)
    validation_frames.append(val_specie)
validation_data = pd.concat(validation_frames)

# Le jeu d'entraînement correspond aux données restantes
training_data = data.drop(validation_data.index)

# Séparation en features et target pour la validation simple
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# Appliquer l'oversampling sur les données d'entraînement
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
# Comptage des effectifs sur les données oversamplées
oversampled_counts = pd.Series(y_train_over).value_counts().sort_index()

# ================================
# 2. Affichage de l'histogramme des effectifs par espèce
# ================================

# Comptage des effectifs dans le jeu complet (pour référence)
full_counts = data[target_col].value_counts().sort_index()

# Comptage dans le jeu d'entraînement et le jeu de validation
train_counts = training_data[target_col].value_counts().sort_index()
val_counts = validation_data[target_col].value_counts().sort_index()

# Trier les espèces par effectifs décroissants dans le training dataset
order = train_counts.sort_values(ascending=False).index
train_counts = train_counts.reindex(order)
val_counts = val_counts.reindex(order, fill_value=0)
oversampled_counts = oversampled_counts.reindex(order, fill_value=0)
additional_counts = oversampled_counts - train_counts

species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
# Partie validation en bas (fixée à 6 par espèce)
plt.bar(x, val_counts, width, color='green', label='Validation dataset (6 per species)')
# Partie training empilée au-dessus
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training dataset')
# Partie oversampling (la différence entre oversampled et training)
plt.bar(x, additional_counts, width, bottom=val_counts + train_counts, color='orange', label='Oversampled training data')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme : Effectifs par espèce (Validation + Training + Oversampling)")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 3. Validation simple (split training / validation)
# ================================

# Entraînement d'un modèle Random Forest sur le jeu d'entraînement oversamplé
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_over, y_train_over)

# Prédictions sur le jeu de validation
y_pred = rf.predict(X_val)

print("=== Validation Simple ===")
print("Classification Report:\n", classification_report(y_val, y_pred))

# Affichage de la matrice de confusion pour la validation simple
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Validation Simple)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("Score de Validation Simple :", accuracy_score(y_val, y_pred))

# ================================
# 4. Validation croisée avec oversampling (via Pipeline)
# ================================

# Utilisation de l'ensemble des données pour la validation croisée
X = data.drop(columns=[target_col])
y = data[target_col]

# Création d'une pipeline intégrant l'oversampling et le modèle Random Forest
pipeline = Pipeline([
    ('oversampler', RandomOverSampler(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Calcul des scores d'accuracy en cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print("Scores de cross-validation (accuracy) avec oversampling:", cv_scores)
print("Accuracy moyenne en cross-validation:", np.mean(cv_scores))

# Prédictions issues de la cross-validation pour afficher le rapport de classification
y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv)
print("=== Rapport de classification (Cross-validation avec oversampling) ===")
print(classification_report(y, y_pred_cv))

# Affichage de la matrice de confusion pour la validation croisée
cm_cv = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(8,6))
sns.heatmap(cm_cv, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Cross-validation avec oversampling)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()
