import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ================================
# 1. Chargement et préparation des données
# ================================

# Charger les données
data = pd.read_csv('C:/Users/alex2/anaconda3/envs/datascience/combined_data.csv', 
                      sep=',', 
                      index_col=0)
print(data)

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

# ================================
# 2. Affichage de l'histogramme des effectifs par espèce
# ================================

# Comptage des effectifs dans l'ensemble des données
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
# 3. Validation simple (split training / validation)
# ================================

# Encodage des étiquettes avec OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  # Remplacer sparse=False par sparse_output=False

# Encodage des labels en One-Hot
y_train_enc = ohe.fit_transform(y_train.values.reshape(-1, 1))  # reshape(-1, 1) si y_train est une série pandas
y_val_enc = ohe.transform(y_val.values.reshape(-1, 1))

# On utilise n_components selon les besoins (ici par exemple 2)
pls = PLSRegression(n_components=2)
pls.fit(X_train, y_train_enc)

y_pred_continuous = pls.predict(X_val)  # Prédictions continues
y_pred_labels = np.argmax(y_pred_continuous, axis=1)  # On prend l'indice de la classe maximale

y_val_labels = np.argmax(y_val_enc, axis=1)  # Extraction des indices de classe des labels one-hot

# Si y_val est encore sous forme de chaîne, on utilise LabelEncoder pour le convertir en entiers
if isinstance(y_val_labels[0], str):
    le = LabelEncoder()
    y_val_labels = le.fit_transform(y_val_labels)

# Rapport de classification
print("=== Validation Simple ===")
print("Classification Report:\n", classification_report(y_val_labels, y_pred_labels))

# Affichage de la matrice de confusion pour la validation simple
cm = confusion_matrix(y_val_labels, y_pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Validation Simple)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

importances = np.abs(pls.coef_).flatten()
plt.figure(figsize=(10,6))
plt.title("Importances des variables (Validation Simple)")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(X_train.columns) // 10)  # <-- baser sur le nb de colonnes réels
positions = list(range(0, len(X_train.columns), step))
labels = X_train.columns[positions]
plt.xticks(positions, labels, rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("Score de Validation Simple :", accuracy_score(y_val_labels, y_pred_labels))
