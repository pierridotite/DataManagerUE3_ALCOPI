import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_score
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ================================
# 1. Chargement et préparation des données
# ================================

data_src = ('/Users/constance/Documents/GitHub/DataManagerUE3_ALCOPI/Data/combined_data.csv')
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

# Création du jeu de validation : 5 individus par espèce
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

# Séparation en features et target
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

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

species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
plt.bar(x, val_counts, width, color='green', label='Validation dataset (5 per species)')
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training dataset')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme : Effectifs par espèce (Training + Validation)")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 3. Entraînement et évaluation du modèle (paramètres par défaut)
# ================================

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

print("Classification Report (Paramètres par défaut):\n", classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Paramètres par défaut)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables (Paramètres par défaut)")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ================================
# 4. Recherche d'hyperparamètres pour le Random Forest avec barre de chargement
# ================================

# Définir une grille d'hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_grid = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_grid,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=0)

n_combinations = len(list(ParameterGrid(param_grid)))
total_iter = n_combinations * 5  # nombre total de tâches (combinaisons x CV folds)
print("\nLancement de la recherche d'hyperparamètres sur {} tâches (combinaisons x CV folds)...".format(total_iter))

with tqdm_joblib(tqdm(desc="GridSearch", total=total_iter)) as progress_bar:
    grid_search.fit(X_train, y_train)

print("\nMeilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)
print("Meilleure accuracy en CV : {:.4f}".format(grid_search.best_score_))

# ================================
# 5. Évaluation du modèle optimisé sur le jeu de validation
# ================================

y_pred_best = grid_search.predict(X_val)

print("\nClassification Report (Modèle optimisé):")
print(classification_report(y_val, y_pred_best))

cm_best = confusion_matrix(y_val, y_pred_best)
plt.figure(figsize=(8,6))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Modèle optimisé)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# ================================
# 6. Validation croisée du modèle optimisé sur l'ensemble complet
# ================================
# Utilisation de l'ensemble complet des données pour évaluer le modèle optimisé

X_full = data.drop(columns=[target_col])
y_full = data[target_col]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Création d'un modèle avec les meilleurs hyperparamètres
best_rf = RandomForestClassifier(random_state=42, **grid_search.best_params_)
cv_scores = cross_val_score(best_rf, X_full, y_full, cv=cv, scoring='accuracy')
print("\nValidation croisée du modèle optimisé - Accuracy moyenne : {:.4f}".format(np.mean(cv_scores)))
