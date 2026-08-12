"""
Road Accident Severity - Model Training

This script:
1. Loads and cleans the accident dataset.
2. Splits the data into training and testing sets.
3. Handles categorical and numerical features.
4. Uses SMOTENC to balance the training data.
5. Trains five different machine-learning models.
6. Evaluates each model.
7. Saves the trained models and useful files for the application.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

from rta_pipeline import SMOTENCFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Basic settings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

BASE = Path(__file__).resolve().parent

MODEL_FOLDER = BASE / "model"
MODEL_FOLDER.mkdir(exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(BASE / "RTA_Dataset.csv")

print("Dataset loaded.")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# These columns contain information about the casualty.
# We are trying to predict accident severity before knowing
# these details, so using them would cause data leakage.

columns_to_remove = [
    "Casualty_severity",
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
]

df = df.drop(columns=columns_to_remove)
missing_values = {
    "unknown",
    "Unknown",
    "na",
    "NA",
    "n/a",
    "",
}

for column in df.columns:

    # Only clean text columns.
    if df[column].dtype == object or str(df[column].dtype).startswith("string"):

        def clean_value(value):

            if pd.isna(value):
                return "Unknown"

            value = str(value).strip()

            if value in missing_values:
                return "Unknown"

            return value

        df[column] = df[column].apply(clean_value)

print("Creating hour_of_day feature...")

df["hour_of_day"] = (
    pd.to_datetime(
        df["Time"],
        format="%H:%M:%S",
        errors="coerce"
    )
    .dt.hour
    .fillna(-1)
    .astype(int)
)

# We do not need the original time anymore.
df = df.drop(columns=["Time"])

TARGET = "Accident_severity"

# These columns have a natural order.
# For example: Below 1 year < 1-2 years < 2-5 years < 5-10 years

ORDINAL_COLUMNS = {

    "Age_band_of_driver": [
        "Under 18",
        "18-30",
        "31-50",
        "Over 51",
        "Unknown",
    ],

    "Driving_experience": [
        "No Licence",
        "Below 1yr",
        "1-2yr",
        "2-5yr",
        "5-10yr",
        "Above 10yr",
        "Unknown",
    ],

    "Service_year_of_vehicle": [
        "Below 1yr",
        "1-2yr",
        "2-5yrs",
        "5-10yrs",
        "Above 10yr",
        "Unknown",
    ],

    "Educational_level": [
        "Illiterate",
        "Writing & reading",
        "Elementary school",
        "Junior high school",
        "High school",
        "Above high school",
        "Unknown",
    ],
}


ordinal_columns = list(ORDINAL_COLUMNS.keys())
ordinal_categories = list(ORDINAL_COLUMNS.values())

numeric_columns = [
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "hour_of_day",
]

# Find nominal categorical columns

nominal_columns = []

for column in df.columns:

    if column not in ordinal_columns:
        if column not in numeric_columns:
            if column != TARGET:
                nominal_columns.append(column)


X = df[ordinal_columns+nominal_columns+numeric_columns].copy()

y_raw = df[TARGET].copy()

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_raw)

class_names = list(label_encoder.classes_)

print("\nTarget classes:")
for number, name in enumerate(class_names):
    print(number, "=", name)

# 80% -> training
# 20% -> testing
#
# The test data is kept separate and untouched.

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

test_data = X_test.copy()

test_data[TARGET] = label_encoder.inverse_transform(y_test)

test_data.to_csv(
    BASE / "test_data.csv",
    index=False
)

print("Saved test_data.csv")

# The ordinal and nominal columns are categorical.
categorical_feature_indices = list(
    range(
        len(ordinal_columns)
        + len(nominal_columns)
    )
)

# Ordinal data:
# Missing values to "Unknown"
# Categories to ordered numbers

ordinal_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )
    ),

    (
        "encoder",
        OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    ),
])


# Nominal data:
# Missing values to "Unknown"
# Categories to one-hot encoded columns

nominal_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    ),
])


# Numerical data:
# Missing values -> median

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
])

preprocessor = ColumnTransformer([

    (
        "ordinal",
        ordinal_pipeline,
        ordinal_columns
    ),

    (
        "nominal",
        nominal_pipeline,
        nominal_columns
    ),

    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
])


models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

def evaluate_model(y_true, y_pred, probabilities):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr",
        average="macro"
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    return {
        "Accuracy": accuracy,
        "AUC (macro OvR)": auc,
        "Precision (macro)": precision,
        "Recall (macro)": recall,
        "F1 (macro)": f1,
        "MCC": mcc,
    }


results = {}

trained_models = {}


for model_name, model in models.items():

    print("Training:", model_name)

    # Build the complete pipeline.
    pipeline = Pipeline([

        (
            "smotenc",
            SMOTENCFrame(
                categorical_features=categorical_feature_indices,
                random_state=RANDOM_STATE,
            )
        ),

        (
            "preprocess",
            preprocessor
        ),

        (
            "scale",
            StandardScaler()
        ),

        (
            "model",
            model
        ),
    ])

    # Train the model.
    pipeline.fit(X_train,y_train)

    # Make predictions on the untouched test data.
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    # Evaluate the model.
    results[model_name] = evaluate_model(y_test,predictions,probabilities)
    # Keep the trained pipeline.
    trained_models[model_name] = pipeline

    # Create a simple filename.
    file_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    model_path = MODEL_FOLDER / (file_name + ".pkl")

    joblib.dump(pipeline,model_path)

    print("Model saved to:", model_path)

results_df = pd.DataFrame(results).T

results_df = results_df[["Accuracy","AUC (macro OvR)","Precision (macro)","Recall (macro)","F1 (macro)","MCC"]].round(4)

print("MODEL COMPARISON")
print(results_df.to_string())

results_df.to_csv(BASE / "model_comparison_results.csv")

per_class_results = []

for model_name, pipeline in trained_models.items():

    probabilities = pipeline.predict_proba(X_test)

    row = {"Model": model_name}
    for class_number, class_name in enumerate(class_names):
        actual_class = (y_test == class_number).astype(int)
        class_auc = roc_auc_score(actual_class,probabilities[:, class_number])
        row["AUC_" + str(class_name)] = round(class_auc,4)
    per_class_results.append(row)

per_class_auc_df = pd.DataFrame(per_class_results).set_index("Model")

print("PER-CLASS AUC")

print(per_class_auc_df.to_string())

per_class_auc_df.to_csv(BASE / "per_class_auc.csv")

joblib.dump(trained_models,BASE / "fitted_pipelines.pkl")

joblib.dump(label_encoder,BASE / "label_encoder.pkl")

feature_metadata = {"ordinal_cols": ordinal_columns, "ordinal_categories": ORDINAL_COLUMNS, "nominal_cols": nominal_columns, "numeric_cols": numeric_columns, "class_names": class_names, "target": TARGET}

with open(BASE / "feature_metadata.json","w") as file:

    json.dump(feature_metadata,file,indent=2)
