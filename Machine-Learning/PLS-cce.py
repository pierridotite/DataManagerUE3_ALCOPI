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

# Load the spectra data
dataviti= pd.read_csv('/Users/constance/Documents/GitHub/DataManagerUE3_ALCOPI/Data/combined_data.csv', 
                      sep=',', 
                      index_col=0)
print(dataviti)

#Préparation des données
# Séparer les variables indépendantes (X) et la variable dépendante (y)
X = dataviti.drop(columns=['species'])
y = dataviti['species']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalisation des données
# Appliquer un standard scaler pour normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Appliquer le fit_transform sur l'ensemble d'entraînement
X_test_scaled = scaler.transform(X_test)        # Appliquer transform sur l'ensemble de test (pas fit)

# Appliquer l'encodage des labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Créer et ajuster le modèle PLS
pls = PLSRegression(n_components=10)
pls.fit(X_train_scaled, y_train_encoded)

#Prédiction et évaluation du modèle 
# Faire des prédictions sur l'ensemble de test
y_pred = pls.predict(X_test_scaled)

# Convertir les prédictions continues en labels (arrondir les valeurs)
y_pred_labels = label_encoder.inverse_transform(np.round(y_pred).astype(int))

#########
# Calculer l'accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_labels)
print("Accuracy: ", accuracy)

# Afficher le rapport de classification
print("Classification Report: ")
print(classification_report(y_test_encoded, y_pred_labels))

# Afficher la matrice de confusion
print("Confusion Matrix: ")
print(confusion_matrix(y_test_encoded, y_pred_labels))

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test_encoded, pls.predict(X_test_scaled))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

