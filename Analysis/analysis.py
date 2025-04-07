#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

# -------------------------------
# 1. Chargement et aperçu du dataset
# -------------------------------
# Remplacer le chemin si nécessaire
data_path = 'C:/Users/alex2/OneDrive/Documents/GitHub/DataManagerUE3_ALCOPI/Analysis/combined_data.csv'
df = pd.read_csv(data_path)

print("Aperçu du dataset :")
print(df.head())
print("\nInformations sur le dataset :")
print(df.info())
print("\nStatistiques descriptives :")
print(df.describe())

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# -------------------------------
# 2. Identification des colonnes spectrales et de la colonne d'espèce
# -------------------------------
# On suppose que la colonne d'espèce s'appelle 'species'. 
# Si ce n'est pas le cas, modifiez la variable ci-dessous.
species_col = 'species'
if species_col not in df.columns:
    print(f"La colonne '{species_col}' n'a pas été trouvée. " \
          "Veuillez modifier le nom de la colonne correspondant aux espèces.")
    species_col = df.columns[-1]  # On suppose ici que c'est la dernière colonne

# Les colonnes spectrales sont toutes les colonnes sauf celle de l'espèce
spectral_columns = [col for col in df.columns if col != species_col]
print("\nColonnes spectrales détectées :")
print(spectral_columns)

# -------------------------------
# 3. Analyse exploratoire
# -------------------------------

# 3.1 Distribution des espèces
species_counts = df[species_col].value_counts()
plt.figure(figsize=(8,6))
plt.bar(species_counts.index, species_counts.values, color='skyblue')
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Distribution des espèces")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.2 Tracé de quelques courbes spectrales individuelles
plt.figure(figsize=(10,6))
# Sélection aléatoire de 10 échantillons (ou moins si le dataset est petit)
sample_df = df.sample(n=min(10, len(df)), random_state=42)
for idx, row in sample_df.iterrows():
    # Conversion des valeurs spectrales en flottant
    spectrum = row[spectral_columns].values.astype(float)
    plt.plot(spectral_columns, spectrum, label=row[species_col])
plt.xlabel("Longueur d'onde")
plt.ylabel("Réflectance / Intensité")
plt.title("Exemples de courbes spectrales")
plt.legend()
# Affichage des x-ticks avec un pas adapté
step = 10 if len(spectral_columns) > 10 else 1
plt.xticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step], rotation=45)
plt.tight_layout()
plt.show()

# 3.3 Courbes moyennes par espèce avec intervalle de confiance (moyenne ± écart-type)
unique_species = df[species_col].unique()
plt.figure(figsize=(10,6))
for species in unique_species:
    subset = df[df[species_col] == species]
    spectra = subset[spectral_columns].astype(float)
    mean_spectrum = spectra.mean()
    std_spectrum = spectra.std()
    wavelengths = spectral_columns  # On suppose que le nom des colonnes représente la longueur d'onde
    plt.plot(wavelengths, mean_spectrum, label=f"{species} (moyenne)")
    plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, alpha=0.2)
plt.xlabel("Longueur d'onde")
plt.ylabel("Réflectance / Intensité")
plt.title("Courbes spectrales moyennes par espèce")
plt.legend()
# Affichage des x-ticks avec un pas adapté
plt.xticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step], rotation=45)
plt.tight_layout()
plt.show()

# 3.4 Calcul et tracé de la première dérivée des courbes moyennes par espèce
plt.figure(figsize=(10,6))
# Paramètres du filtre Savitzky-Golay (à ajuster selon votre résolution)
window_length = 7  # doit être impair
polyorder = 2
for species in unique_species:
    subset = df[df[species_col] == species]
    spectra = subset[spectral_columns].astype(float)
    mean_spectrum = spectra.mean().values
    # Vérification que window_length est adapté
    if window_length > len(mean_spectrum):
        window_length = len(mean_spectrum) if len(mean_spectrum) % 2 != 0 else len(mean_spectrum)-1
    first_derivative = savgol_filter(mean_spectrum, window_length=window_length, polyorder=polyorder, deriv=1)
    plt.plot(spectral_columns, first_derivative, label=f"{species} (dérivée)")
plt.xlabel("Longueur d'onde")
plt.ylabel("Dérivée de la réflectance")
plt.title("Première dérivée des courbes spectrales moyennes par espèce")
plt.legend()
# Affichage des x-ticks avec un pas adapté
plt.xticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step], rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Analyse en Composantes Principales (PCA)
# -------------------------------
# Application de la PCA sur les données spectrales
spectra_data = df[spectral_columns].astype(float)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(spectra_data)
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Nouveaux logs pour afficher les résultats de la PCA
print("\nRésultats de la PCA :")
print("Rapport de variance expliquée : ", pca.explained_variance_ratio_)
print("Aperçu de la projection PCA (5 premières lignes) :\n", df[['PC1','PC2']].head())

plt.figure(figsize=(8,6))
for species in unique_species:
    subset = df[df[species_col] == species]
    plt.scatter(subset['PC1'], subset['PC2'], label=species)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection PCA des données spectrales")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Matrice de corrélation des bandes spectrales
# -------------------------------
corr_matrix = spectra_data.corr()
plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Matrice de corrélation des bandes spectrales")
# Affichage des x-ticks avec un pas adapté pour éviter le chevauchement
plt.xticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step], rotation=90)
plt.yticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step])
plt.tight_layout()
plt.show()
