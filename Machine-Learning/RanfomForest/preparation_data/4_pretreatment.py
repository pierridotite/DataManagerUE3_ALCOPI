import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.signal import savgol_filter
from sklearn.pipeline import make_pipeline

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
# Partie validation en bas (fixée à 5 par espèce)
plt.bar(x, val_counts, width, color='green', label='Validation dataset (5 per species)')
# Partie training empilée au-dessus
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training dataset')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme : Effectifs par espèce (Training + Validation)")
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 3. Entraînement et évaluation du modèle de base (sans prétraitement supplémentaire)
# ================================

# Entraînement d'un modèle Random Forest sur le jeu d'entraînement de base
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Prédictions sur le jeu de validation
y_pred = rf.predict(X_val)

# Affichage du rapport de classification
print("=== Modèle sans prétraitement supplémentaire ===")
print("Classification Report:\n", classification_report(y_val, y_pred))

# Affichage de la matrice de confusion
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion (Aucun prétraitement)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Calcul du R² (après encodage des labels)
le = LabelEncoder()
y_val_enc = le.fit_transform(y_val)
y_pred_enc = le.transform(y_pred)
r2 = r2_score(y_val_enc, y_pred_enc)
print("R² :", r2)

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables (Aucun prétraitement)")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ================================
# 4. Définition des fonctions de prétraitement
# ================================

def identity(X):
    """Aucun prétraitement."""
    return X.copy()

def snv(X):
    """Application de la transformation SNV (Standard Normal Variate) sur chaque ligne."""
    X_array = X.values
    X_snv_array = np.array([ (row - row.mean())/row.std() if row.std() != 0 else row - row.mean() for row in X_array])
    return pd.DataFrame(X_snv_array, index=X.index, columns=X.columns)

def first_derivative(X):
    """Calcul de la 1ère dérivée pour chaque échantillon."""
    X_array = X.values
    X_deriv = np.gradient(X_array, axis=1)
    return pd.DataFrame(X_deriv, index=X.index, columns=X.columns)

def second_derivative(X):
    """Calcul de la 2ème dérivée pour chaque échantillon."""
    X_array = X.values
    first_deriv = np.gradient(X_array, axis=1)
    second_deriv = np.gradient(first_deriv, axis=1)
    return pd.DataFrame(second_deriv, index=X.index, columns=X.columns)

def snv_first_derivative(X):
    """SNV suivi de la 1ère dérivée."""
    return first_derivative(snv(X))

def snv_second_derivative(X):
    """SNV suivi de la 2ème dérivée."""
    return second_derivative(snv(X))

def savitzky_golay(X, window_length=7, polyorder=2):
    """Filtrage Savitzky-Golay pour le lissage des données."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), 1, X.values)
    return pd.DataFrame(X_sg, index=X.index, columns=X.columns)

def savgol_first_derivative(X, window_length=7, polyorder=2):
    """Filtrage Savitzky-Golay suivi du calcul de la 1ère dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg1 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=1), 1, X.values)
    return pd.DataFrame(X_sg1, index=X.index, columns=X.columns)

def savgol_second_derivative(X, window_length=7, polyorder=2):
    """Filtrage Savitzky-Golay suivi du calcul de la 2ème dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg2 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=2), 1, X.values)
    return pd.DataFrame(X_sg2, index=X.index, columns=X.columns)

def standard_scaling_train_val(X_train, X_val):
    """Standardisation column-wise : mise à l'échelle par moyenne et variance (appliquée avec ajustement sur le training set)."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    return X_train_scaled, X_val_scaled

# ================================
# 5. Dictionnaire des méthodes de prétraitement à tester
# ================================

preprocessing_methods = {
    "Aucun": identity,
    "SNV": snv,
    "1ère dérivée": first_derivative,
    "2ème dérivée": second_derivative,
    "SNV + 1ère dérivée": snv_first_derivative,
    "SNV + 2ème dérivée": snv_second_derivative,
    "Savitzky-Golay": savitzky_golay,
    "Savitzky-Golay + 1ère dérivée": savgol_first_derivative,
    "Savitzky-Golay + 2ème dérivée": savgol_second_derivative,
    "Standard Scaling": None  # Traitement particulier
}

# ================================
# 6. Test de chaque prétraitement et comparaison des performances (validation simple)
# ================================

results = {}  # Pour stocker accuracy et autres métriques

for method_name, func in preprocessing_methods.items():
    print(f"\n=== Prétraitement: {method_name} ===")
    # Gestion particulière pour Standard Scaling (ajustement sur X_train)
    if method_name == "Standard Scaling":
        X_train_trans, X_val_trans = standard_scaling_train_val(X_train, X_val)
    else:
        X_train_trans = func(X_train)
        X_val_trans = func(X_val)
    
    # Entraînement du modèle
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_trans, y_train)
    y_pred = rf.predict(X_val_trans)
    
    # Calcul des métriques
    acc = accuracy_score(y_val, y_pred)
    results[method_name] = {"accuracy": acc}
    
    # Affichage du rapport de classification
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion ({method_name})")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables")
    plt.tight_layout()
    plt.show()

# Création d'un DataFrame des résultats et affichage
results_df = pd.DataFrame(results).T
print("\nComparaison des métriques :")
print(results_df)

# ================================
# 7. Visualisation comparée des métriques (accuracy)
# ================================

plt.figure(figsize=(8,4))
plt.bar(results_df.index, results_df['accuracy'], color='skyblue')
plt.ylabel("Accuracy")
plt.title("Comparaison des accuracies selon le prétraitement")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# 8. Validation croisée pour chaque prétraitement
# ================================
# On utilise l'ensemble complet des données pour la validation croisée
X_full = data.drop(columns=[target_col])
y_full = data[target_col]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for method_name, func in preprocessing_methods.items():
    if method_name == "Standard Scaling":
        # Pour Standard Scaling, on utilise le StandardScaler de sklearn
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    elif method_name == "Aucun":
        pipeline = make_pipeline(RandomForestClassifier(random_state=42))
    else:
        # Pour les autres, on utilise un FunctionTransformer avec validate=False pour conserver le DataFrame
        transformer = FunctionTransformer(func, validate=False)
        pipeline = make_pipeline(transformer, RandomForestClassifier(random_state=42))
    
    scores = cross_val_score(pipeline, X_full, y_full, cv=cv, scoring='accuracy')
    cv_results[method_name] = np.mean(scores)
    print(f"Validation croisée - Méthode: {method_name}, Accuracy moyenne: {np.mean(scores):.4f}")

cv_results_df = pd.DataFrame.from_dict(cv_results, orient='index', columns=['CV Accuracy'])
plt.figure(figsize=(8,4))
plt.bar(cv_results_df.index, cv_results_df['CV Accuracy'], color='lightgreen')
plt.xticks(rotation=45)
plt.ylabel("CV Accuracy")
plt.title("Validation croisée des prétraitements")
plt.tight_layout()
plt.show()
