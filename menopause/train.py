# Import required libraries
import pandas as pd            # For handling dataset
import pickle                  # For saving models
import os  
import numpy as np                    # For file/folder operations

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create folder to store trained models
os.makedirs("models", exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------



# Load dataset
df = pd.read_csv("menopause_dataset.csv")

# -----------------------------
# ADD AMH_Level (based on age)
# -----------------------------
def generate_amh(age):
    if age < 30:
        return np.random.uniform(2.5, 4.0)
    elif age < 40:
        return np.random.uniform(1.5, 2.5)
    elif age < 45:
        return np.random.uniform(0.8, 1.5)
    else:
        return np.random.uniform(0.1, 0.8)

df["AMH_Level"] = df["Age"].apply(generate_amh)

# -----------------------------
# CREATE Anxiety_Level
# -----------------------------
def map_anxiety(val):
    if val == "Yes":
        return np.random.choice([1, 2, 3])  # Mild to Severe
    else:
        return 0

df["Anxiety_Level"] = df["Anxiety"].apply(map_anxiety)



print("✅ Dataset updated with AMH_Level and Anxiety_Level")

# Shuffle dataset (important to avoid bias)
df = df.sample(frac=1, random_state=42)

# Import XGBoost (powerful ML model)
from xgboost import XGBClassifier

# -----------------------------
# ENCODE YES/NO → 0/1
# -----------------------------
yes_no_cols = [
    "Irregular_Periods", "Missed_Periods", "Hot_Flashes",
    "Night_Sweats", "Sleep_Problems",
    "Vaginal_Dryness", "Joint_Pain"
]

# Convert Yes → 1, No → 0
for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# -----------------------------
# CREATE TARGET LABEL (Menopause Stage)
# -----------------------------
def assign_stage(row):
    # Post-menopause (no period for 1+ year)
    if row["Years_Since_Last_Period"] >= 1:
        return "Post-Menopause"

    # Strong indicator using AMH + Age
    elif row["AMH_Level"] < 0.5 and row["Age"] >= 45:
        return "Peri-Menopause"

    # General symptoms + age condition
    elif row["Age"] >= 40 and (
        row["Irregular_Periods"] == 1 or
        row["Hot_Flashes"] == 1
    ):
        return "Peri-Menopause"

    # Otherwise normal reproductive stage
    else:
        return "Pre-Menopause"

# Apply rule to create label column
df["Menopause_Stage"] = df.apply(assign_stage, axis=1)

# -----------------------------
# ADD NEW FEATURES (AMH + Anxiety)
# -----------------------------

# Convert anxiety text → numeric
anxiety_map = {
    "None": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3
}

# Ensure AMH is numeric (handle wrong values)
df["AMH_Level"] = pd.to_numeric(df["AMH_Level"], errors="coerce")

# Fill missing AMH with median value
df["AMH_Level"] = df["AMH_Level"].fillna(df["AMH_Level"].median())



# Fill missing anxiety with 0 (None)
df["Anxiety_Level"] = df["Anxiety_Level"].fillna(0)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Convert AMH into categories (helps model learn patterns)
def amh_category(val):
    if val > 2:
        return 0   # High
    elif val > 1:
        return 1   # Normal
    elif val > 0.5:
        return 2   # Low
    else:
        return 3   # Very Low

df["AMH_Category"] = df["AMH_Level"].apply(amh_category)

# Combine AMH + Anxiety into single score
def hormonal_score(row):
    score = 0

    # AMH contribution
    if row["AMH_Level"] < 1:
        score += 2
    if row["AMH_Level"] < 0.5:
        score += 3

    # Anxiety contribution
    if row["Anxiety_Level"] >= 2:
        score += 1
    if row["Anxiety_Level"] == 3:
        score += 2

    return score

df["Hormonal_Transition_Score"] = df.apply(hormonal_score, axis=1)

# -----------------------------
# SELECT FEATURES (INPUT DATA)
# -----------------------------
features = [
    "Age",
    "Estrogen_Level",
    "FSH_Level",
    "AMH_Level",
    "AMH_Category",
    "Anxiety_Level",
    "Hormonal_Transition_Score",
    "Years_Since_Last_Period",
    "Irregular_Periods",
    "Missed_Periods",
    "Hot_Flashes",
    "Night_Sweats",
    "Sleep_Problems",
    "Vaginal_Dryness",
    "Joint_Pain"
]

# Input features (X) and target (y)
X = df[features]
y = df["Menopause_Stage"]

# -----------------------------
# ENCODE TARGET LABELS
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# SCALING (ONLY FOR KNN)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# TRAIN MODELS
# -----------------------------

# XGBoost (advanced model)
xgb = XGBClassifier(n_estimators=200, random_state=42)
xgb.fit(X_train, y_train)

# Random Forest (main model)
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

# KNN (for recommendation/similarity)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
print("RF Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print("XGB Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))
print("KNN Accuracy:", accuracy_score(y_test, knn.predict(X_test_scaled)))

# -----------------------------
# SAVE MODELS (VERY IMPORTANT)
# -----------------------------
pickle.dump(rf, open("models/rf_model.pkl", "wb"))
pickle.dump(xgb, open("models/xgb_model.pkl", "wb"))
pickle.dump(knn, open("models/knn_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

print("✅ All models saved successfully")

#save dataset
df.to_csv("final_dataset.csv", index=False)