import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Charger les données
data = pd.read_csv(r'Data\combined_data.csv')
original_counts = data.iloc[:, -1].value_counts()  # Sauvegarde des données de base

# Shuffle and split data into 70% training and 30% test (drop remaining rows)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(data)
train_count = int(0.70 * n)
test_count = int(0.30 * n)
train_data = data.iloc[:train_count].copy()
test_data = data.iloc[train_count:train_count+test_count].copy()



label_col = data.columns[-1]
species_list = train_data[label_col].unique()
max_count = train_data[label_col].value_counts().max()

balanced_extra_list = []
noise_augmented_list = []

for sp in species_list:
    sp_data = train_data[train_data[label_col] == sp]
    current_count = len(sp_data)
    dup_needed = max_count - current_count
    if dup_needed > 0:
        # Duplicate extra rows (balanced augmentation) only if needed
        extra = sp_data.sample(n=dup_needed, replace=True, random_state=42)
        balanced_extra_list.append(extra)
    # Always generate noise-augmented rows for all species
    num_cols = sp_data.select_dtypes(include=[np.number]).columns
    sp_min = sp_data[num_cols].min()
    sp_max = sp_data[num_cols].max()
    # Generate noise-augmented rows with count equal to max_count for each species.
    noise_rows = pd.DataFrame({
        col: np.random.uniform(sp_min[col], sp_max[col], size=max_count)
        for col in num_cols
    })
    noise_rows[label_col] = sp  # set species label
    noise_augmented_list.append(noise_rows)

balanced_extra = pd.concat(balanced_extra_list) if balanced_extra_list else pd.DataFrame(columns=train_data.columns)
noise_augmented = pd.concat(noise_augmented_list) if noise_augmented_list else pd.DataFrame(columns=train_data.columns)

# Plot stacked bar chart for each species:
# Test data (red), Base training (blue), Balanced extra (orange), Noise-augmented (green)
bar_df = pd.DataFrame({
    'Species': species_list,
    'Test': [len(test_data[test_data[label_col] == sp]) for sp in species_list],
    'Base Train': [len(train_data[train_data[label_col] == sp]) for sp in species_list],
    'Balanced Extra': [len(balanced_extra[balanced_extra[label_col] == sp]) for sp in species_list],
    'Noise Augmented': [len(noise_augmented[noise_augmented[label_col] == sp]) for sp in species_list]
})
bar_df.set_index('Species', inplace=True)
bar_df.plot(kind='bar', stacked=True, color=['red','blue','orange','green'])
plt.ylabel('Count')
plt.title('Data Composition per Species')
plt.tight_layout()
plt.show()

# Use base training + noise augmented rows for training
balanced_train = pd.concat([train_data, noise_augmented]).reset_index(drop=True)
X_train = balanced_train.drop(columns=[label_col])
y_train = balanced_train[label_col]
X_test = test_data.drop(columns=[label_col])
y_test = test_data[label_col]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForestClassifier training and evaluation
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Graphique des importances des variables
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.title("Importances des variables")
plt.bar(range(len(importances)), importances, align="center")
plt.tight_layout()
plt.show()

# Compute R² score after encoding species labels numerically
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test_num = le.fit_transform(y_test)
y_pred_num = le.transform(y_pred)
r2 = r2_score(y_test_num, y_pred_num)
print("R² score:", r2)
