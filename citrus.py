# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# %%
# Read dataset
df = pd.read_csv("citrus.csv")

print(df.head())
print(df.info())

# %%
# Encode label (orange/grape)
le = LabelEncoder()
df["name"] = le.fit_transform(df["name"])  
# grape=0, orange=1 (or reverse depending on data order)

# %%
# Features and target
X = df.drop("name", axis=1)
y = df["name"]

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ==================================================
# 1. DECISION TREE
# ==================================================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))

# ==================================================
# 2. NAIVE BAYES
# ==================================================
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print("=== Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))

# ==================================================
# 3. SUPPORT VECTOR MACHINE
# ==================================================
svm = SVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("=== SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# %%
# Confusion Matrix Heatmap

# ==================================================
# 1. DECISION TREE
# ==================================================
cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==================================================
# 2. NAIVE BAYES
# ==================================================
cm = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==================================================
# 3. SUPPORT VECTOR MACHINE
# ==================================================
cm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#%%
# Accuracy Comparison Bar Chart
models = ['Decision Tree', 'Naive Bayes', 'SVM']
scores = [0.9384, 0.9208, 0.9256]

plt.figure(figsize=(8,5))
bars = plt.bar(models, scores)

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, y+0.001, round(y,4), ha='center')

plt.ylim(0.88,1.0)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()