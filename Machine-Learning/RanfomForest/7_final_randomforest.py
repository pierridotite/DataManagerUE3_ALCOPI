import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter

# ================================
# 1. Chargement et préparation des données
# ================================

data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

# Création du jeu de validation : 6 individus par espèce (avec ou sans remise selon la disponibilité)
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
# 2. Définition des fonctions de prétraitement
# ================================

def snv(X):
    """
    Application de la transformation SNV (Standard Normal Variate) sur chaque ligne.
    """
    X_array = X.values
    X_snv_array = np.array([
        (row - row.mean()) / row.std() if row.std() != 0 else row - row.mean()
        for row in X_array
    ])
    return pd.DataFrame(X_snv_array, index=X.index, columns=X.columns)

def first_derivative(X):
    """
    Calcul de la première dérivée pour chaque échantillon.
    """
    X_array = X.values
    X_deriv = np.gradient(X_array, axis=1)
    return pd.DataFrame(X_deriv, index=X.index, columns=X.columns)

def snv_first_derivative(X):
    """
    Application de SNV suivie du calcul de la première dérivée.
    """
    return first_derivative(snv(X))

# ================================
# 3. Application du prétraitement SNV + 1ère dérivée
# ================================

X_train_trans = snv_first_derivative(X_train)
X_val_trans = snv_first_derivative(X_val)

# ================================
# 4. Entraînement et évaluation du modèle Random Forest
# ================================

# Entraînement du modèle
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_trans, y_train)

# Prédictions sur le jeu de validation
y_pred = rf.predict(X_val_trans)

# Affichage du rapport de classification
print("=== Random Forest avec SNV + 1ère dérivée ===")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Calcul et affichage de l'accuracy
acc = accuracy_score(y_val, y_pred)
print("Accuracy:", acc)

# Affichage de la matrice de confusion
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (SNV + 1ère dérivée)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# ================================
# 5. Validation croisée avec SNV + 1ère dérivée
# ================================
# Nous intégrons le prétraitement dans un pipeline pour effectuer la validation croisée sur l'ensemble complet.
# On utilise un FunctionTransformer pour appliquer snv_first_derivative.

# Création du pipeline
pipeline = make_pipeline(FunctionTransformer(snv_first_derivative, validate=False),
                           RandomForestClassifier(random_state=42))

# Utilisation de StratifiedKFold pour conserver la répartition des classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, data.drop(columns=[target_col]), data[target_col], cv=cv, scoring='accuracy')

print("\nValidation croisée avec SNV + 1ère dérivée")
print("CV Accuracy scores:", cv_scores)
print("CV Accuracy moyenne:", np.mean(cv_scores))
