import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid

from imblearn.over_sampling import RandomOverSampler

from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from scipy.signal import savgol_filter

# =============================================================================
# Fonctions de prétraitement (issues de pretreatment.py)
# =============================================================================

def identity(X):
    """Aucun prétraitement."""
    return X.copy()

def snv(X):
    """Transformation SNV (Standard Normal Variate) appliquée ligne par ligne."""
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
    """Filtrage Savitzky-Golay pour le lissage."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), 1, X.values)
    return pd.DataFrame(X_sg, index=X.index, columns=X.columns)

def savgol_first_derivative(X, window_length=7, polyorder=2):
    """Savitzky-Golay suivi du calcul de la 1ère dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg1 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=1), 1, X.values)
    return pd.DataFrame(X_sg1, index=X.index, columns=X.columns)

def savgol_second_derivative(X, window_length=7, polyorder=2):
    """Savitzky-Golay suivi du calcul de la 2ème dérivée."""
    n_cols = X.shape[1]
    if window_length > n_cols:
        window_length = n_cols if n_cols % 2 == 1 else n_cols - 1
    X_sg2 = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder, deriv=2), 1, X.values)
    return pd.DataFrame(X_sg2, index=X.index, columns=X.columns)

def standard_scaling_train_val(X_train, X_val):
    """Standardisation column-wise (ajustement sur X_train)."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    return X_train_scaled, X_val_scaled

# Dictionnaire des méthodes de prétraitement disponibles
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

# Choix de la méthode de prétraitement (modifiable)
preprocessing_method = "Standard Scaling"

# =============================================================================
# Chargement et séparation des données
# =============================================================================

data_src = r'Data\combined_data.csv'
data = pd.read_csv(data_src)

# On considère que la dernière colonne est la cible
target_col = data.columns[-1]

# Création du jeu de validation : 6 individus par espèce
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

# Séparation en features et cible
X_train = training_data.drop(columns=[target_col])
y_train = training_data[target_col]
X_val = validation_data.drop(columns=[target_col])
y_val = validation_data[target_col]

# =============================================================================
# Visualisation des effectifs par espèce (Training + Validation)
# =============================================================================

train_counts = training_data[target_col].value_counts().sort_index()
val_counts = validation_data[target_col].value_counts().sort_index()
order = train_counts.sort_values(ascending=False).index
train_counts = train_counts.reindex(order)
val_counts = val_counts.reindex(order, fill_value=0)
species = order
x = np.arange(len(species))
width = 0.6

plt.figure(figsize=(10, 6))
plt.bar(x, val_counts, width, color='green', label='Validation (6 par espèce)')
plt.bar(x, train_counts, width, bottom=val_counts, color='blue', label='Training')
plt.xticks(x, species, rotation=45)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme des effectifs par espèce")
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# Prétraitement des données
# =============================================================================

if preprocessing_method == "Standard Scaling":
    X_train_proc, X_val_proc = standard_scaling_train_val(X_train, X_val)
else:
    func = preprocessing_methods[preprocessing_method]
    X_train_proc = func(X_train)
    X_val_proc = func(X_val)

# =============================================================================
# Oversampling pour équilibrer les classes (RandomOverSampler)
# =============================================================================

ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_proc, y_train)

# =============================================================================
# Recherche d'hyperparamètres avec GridSearchCV
# =============================================================================

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=0)

n_combinations = len(list(ParameterGrid(param_grid)))
total_iter = n_combinations * 5  # 5 folds de CV
print("\nRecherche d'hyperparamètres sur {} combinaisons...".format(total_iter))
with tqdm_joblib(tqdm(desc="GridSearch", total=total_iter)) as progress_bar:
    grid_search.fit(X_train_bal, y_train_bal)

print("\nMeilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)
print("Meilleure accuracy en CV : {:.4f}".format(grid_search.best_score_))

# =============================================================================
# Entraînement du modèle final avec les meilleurs hyperparamètres
# =============================================================================

final_model = grid_search.best_estimator_
final_model.fit(X_train_bal, y_train_bal)

# =============================================================================
# Évaluation sur le jeu de validation
# =============================================================================

y_pred = final_model.predict(X_val_proc)
print("\nRapport de classification sur le jeu de validation :")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.tight_layout()
plt.show()

# Calcul du R² (après encodage des labels)
le = LabelEncoder()
y_val_enc = le.fit_transform(y_val)
y_pred_enc = le.transform(y_pred)
r2 = r2_score(y_val_enc, y_pred_enc)
print("R² sur le jeu de validation :", r2)

# =============================================================================
# Visualisation des importances des variables
# =============================================================================

importances = final_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X_train.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
