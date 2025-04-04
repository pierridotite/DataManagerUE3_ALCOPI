import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Charger les données
data = pd.read_csv(r'c:\Users\ACER\OneDrive\Bureau\DataManagerUE3_ALCOPI\Data\combined_data.csv')
# On considère que la dernière colonne est la cible
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modèle Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Affichage des résultats dans la console
accuracy = accuracy_score(y_test, y_pred)
print("Précision :", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)
print(classification_report(y_test, y_pred))

# Plot de la matrice de confusion
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.show()

# Validation supplémentaire pour data analyst : calcul de R carré
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test_enc = le.fit_transform(y_test)
y_pred_enc = le.transform(y_pred)
r2 = r2_score(y_test_enc, y_pred_enc)
print("R carré :", r2)

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
# Limiter le nombre de labels affichés sur l'axe x
step = max(1, len(importances) // 10)
plt.xticks(range(0, len(importances), step), X.columns[::step], rotation=45, ha="right")
plt.tight_layout()
plt.show()
