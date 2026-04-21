from google.colab import drive
drive.mount('/content/drive')

!pip install imbalanced-learn -q

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

print("✅ Ready!")

DRIVE_PATH = '/content/drive/MyDrive/UML_Project/'

with h5py.File(DRIVE_PATH + 'semantic_embeddings.h5', 'r') as f:
    classification_emb = f['classification_embedding'][:]
    urls = f['URL'][:]

urls = [u.decode('utf-8') if isinstance(u, bytes) else u for u in urls]
print(f"✅ Embeddings shape: {classification_emb.shape}")
print(f"✅ URLs count: {len(urls)}")

df = pd.read_csv(DRIVE_PATH + 'EDA_data-FULL.csv')

valid_labels = ['update-me', 'give-me-perspective', 'educate-me', 'connect-me', 'inspire-me', 'help-me']
df_filtered = df[df['User_Needs'].isin(valid_labels)].copy().reset_index(drop=True)

emb_df = pd.DataFrame({'URL': urls, 'emb_index': range(len(urls))})
merged = emb_df.merge(df_filtered[['URL', 'User_Needs']], on='URL', how='inner')

print(f"✅ Matched: {len(merged)} articles")
print(merged['User_Needs'].value_counts())

X = classification_emb[merged['emb_index'].values]
y_raw = merged['User_Needs'].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

classifiers = {
    "SVM (RBF)": SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    "MLP Neural Net": MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        max_iter=300,
        learning_rate_init=0.001,
        early_stopping=True,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=10.0,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42
    )
}

best_acc = 0
best_name = ""
best_pred = None

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_scaled, y_train_bal)
    pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    print(f"  → Accuracy: {acc*100:.1f}%")
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_pred = pred

print(f"\n{'='*50}")
print(f"  BEST: {best_name}")
print(f"  ACCURACY: {best_acc*100:.1f}%")
print(f"{'='*50}\n")
print(classification_report(y_test, best_pred, target_names=le.classes_))
