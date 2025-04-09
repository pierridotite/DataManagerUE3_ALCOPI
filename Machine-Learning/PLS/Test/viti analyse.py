# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from chemotools.derivative import SavitzkyGolay
from chemotools.scatter import StandardNormalVariate
from chemotools.feature_selection import RangeCut
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 
from sklearn.cross_decomposition import PLSRegression

### CHARGEMENT DES DONNEES ###

# Load the combined data
dataviti = pd.read_csv('Data\combined_data.csv', 
                      sep=',', 
                      index_col=0)
print(dataviti)

### SEPARATION DES VARIABLES EXPLICATIVES (SPECTRE) ET CIBLE (CLASS) ###

# Séparation des variables explicatives et cible
X = dataviti.drop(columns=["class"])  # Supprime la colonne cible pour garder les variables explicatives
y = dataviti["class"]  # Récupère la colonne cible

# Encodage de la variable cible (transforme les classes en nombres)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Vérifier les classes encodées
print("Classes encodées :", label_encoder.classes_)

### NORMALISATION DES DONNEES ###

# Standardisation des variables explicatives
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Vérifier la moyenne et l'écart-type après standardisation
print("Moyenne après standardisation :", X_scaled.mean(axis=0).round(2))
print("Écart-type après standardisation :", X_scaled.std(axis=0).round(2))

### MODELISATION ###

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

# Tester plusieurs nombres de composantes
n_components_list = range(1, 21)
scores = []

for n in n_components_list:
    pls = PLSRegression(n_components=n)
    score = cross_val_score(pls, X_train, y_train, cv=5, scoring='r2').mean()
    scores.append(score)

# Trouver le nombre optimal de composantes
best_n_components = n_components_list[np.argmax(scores)]
print(f"Nombre optimal de composantes : {best_n_components}")

# Entraînement du modèle avec le meilleur nombre de composantes
pls_final = PLSRegression(n_components=best_n_components)
pls_final.fit(X_train, y_train)

# Prédiction sur le test
y_pred = pls_final.predict(X_test)

# Évaluer la performance
print("Score R2 :", pls_final.score(X_test, y_test))

### OPTIMISATION DU MODELE ###

def interpoler(X, y, classe, n_samples=20):
    """ Crée des nouvelles données en interpolant entre deux échantillons existants """
    X_class = X[y == classe]
    indices = np.random.randint(0, len(X_class), (n_samples, 2))
    X_interp = (X_class[indices[:, 0]] + X_class[indices[:, 1]]) / 2
    y_interp = np.array([classe] * n_samples)

 #Visualisation des données originales et des nouvelles données générées
    plt.figure(figsize=(8, 6))
    
    # Points originaux (en bleu)
    plt.scatter(X_class[:, 0], X_class[:, 1], color='blue', label='Données originales', alpha=0.5)
    
    # Points générés (en rouge)
    plt.scatter(X_interp[:, 0], X_interp[:, 1], color='red', label='Données interpolées', alpha=0.7)
    
    plt.title(f'Visualisation de l\'interpolation pour la classe {classe}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right')
    
    plt.show()

    return X_interp, y_interp

