import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# ================================
# 1. Chargement et préparation des données
# ================================

# Charger les données
data_src = r'Data\combined_data.csv'
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

# Appliquer l'oversampling sur les données d'entraînement
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
# Comptage des effectifs sur les données oversamplées
oversampled_counts = pd.Series(y_train_over).value_counts().sort_index()

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
oversampled_counts = oversampled_counts.reindex(order, fill_value=0)
additional_counts = oversampled_counts - train_counts

species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10,6))
# Partie validation en bas (fixée à 5 par espèce)
plt.bar(x, val_counts, width, color='green', label='Validation dataset (5 per species)')
# Partie training empilée au-dessus
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training dataset')
# Partie oversampling au-dessus (la différence entre oversampled et training)
plt.bar(x, additional_counts, width, bottom=val_counts + train_counts, color='orange', label='Oversampled training data')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme : Effectifs par espèce (Validation + Training + Oversampling)")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 3. Entraînement et évaluation du modèle
# ================================

# Encodage des classes en chiffres pour PLS
le = LabelEncoder()
y_train_over_encoded = le.fit_transform(y_train_over)

# Entraînement d'un modèle PLS + Logistic Regression sur le jeu d'entraînement oversamplé
pls = PLSRegression(n_components=5)  # Choisis le nombre de composantes selon ton problème
X_train_over_pls = pls.fit_transform(X_train_over, y_train_over_encoded)[0]
X_val_pls = pls.transform(X_val)

# Logistic Regression sur les composantes PLS
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_over_pls, y_train_over)

# Prédictions sur le jeu de validation
y_pred = lr.predict(X_val_pls)

# Affichage du rapport de classification
print("Classification Report:\n", classification_report(y_val, y_pred))

# Affichage de la matrice de confusion
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Graphique des importances des variables
importances = np.sum(np.abs(pls.x_weights_), axis=1)
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()