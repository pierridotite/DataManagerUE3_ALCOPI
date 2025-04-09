import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ================================
# 1. Chargement et préparation des données
# ================================

# Charger les données
data = pd.read_csv('C:/Users/alex2/anaconda3/envs/datascience/combined_data.csv', 
                   sep=',', 
                   index_col=0)

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

training_data = data.drop(validation_data.index)

# Séparation en features et target
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]

X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# Standardisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Encodage des labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

# ================================
# 2. Affichage de l'histogramme des effectifs par espèce
# ================================

full_counts = data[target_col].value_counts().sort_index()
train_counts = training_data[target_col].value_counts().sort_index()
val_counts = validation_data[target_col].value_counts().sort_index()

order = train_counts.sort_values(ascending=False).index
train_counts = train_counts.reindex(order)
val_counts = val_counts.reindex(order, fill_value=0)

species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
plt.bar(x, val_counts, width, color='green', label='Validation dataset (6 per species)')
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training dataset')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme : Effectifs par espèce (Training + Validation)")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 3. Entraînement et évaluation du modèle PLS (paramètres par défaut)
# ================================

# Définir un modèle PLS simple avec un nombre fixe de composantes
pls = OneVsRestClassifier(PLSRegression(n_components=2))
pls.fit(X_train_scaled, y_train_enc)

# Prédictions
y_pred_enc = pls.predict(X_val_scaled)
y_pred_enc = np.round(y_pred_enc).astype(int).ravel()  # Argmax car pls.predict donne des valeurs continues

# Décodage des labels prédits
y_pred = le.inverse_transform(y_pred_enc)

print("\nClassification Report (Paramètres par défaut):")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Paramètres par défaut)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Calcul du R²
r2 = r2_score(y_val_enc, y_pred_enc)
print("R² :", r2)

# ================================
# 4. Recherche du meilleur nombre de composantes pour le modèle PLS
# ================================

# Définir la grille de recherche pour n_components
param_grid = {'estimator__n_components': list(range(2, min(30, X_train.shape[1])))}

pls_grid = OneVsRestClassifier(PLSRegression())
grid_search = GridSearchCV(pls_grid, param_grid, cv=5, n_jobs=-1, scoring='accuracy')

total_iter = len(param_grid['estimator__n_components']) * 5
print("\nLancement de la recherche du meilleur nombre de composantes sur {} tâches...".format(total_iter))

with tqdm_joblib(tqdm(desc="GridSearch PLS", total=total_iter)):
    grid_search.fit(X_train_scaled, y_train_enc)

print("\nMeilleur nombre de composantes trouvé :", grid_search.best_params_['estimator__n_components'])
print("Meilleure accuracy en CV : {:.4f}".format(grid_search.best_score_))

# ================================
# 5. Évaluation du modèle optimisé
# ================================

y_pred_best_enc = grid_search.predict(X_val_scaled)
y_pred_best_enc = np.argmax(y_pred_best_enc, axis=1)

y_pred_best = le.inverse_transform(y_pred_best_enc)

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
