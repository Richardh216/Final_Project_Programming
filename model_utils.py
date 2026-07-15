"""Shared data preparation, model comparison, and training utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path(__file__).with_name("dynamic_pricing.csv")
TARGET = "Historical_Cost_of_Ride"

NUMERIC_COLUMNS = [
    "Number_of_Riders",
    "Number_of_Drivers",
    "Number_of_Past_Rides",
    "Average_Ratings",
    "Expected_Ride_Duration",
    TARGET
]

CATEGORICAL_COLUMNS = [
    "Vehicle_Type",
    "Location_Category",
    "Time_of_Booking",
    "Customer_Loyalty_Status"
]
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy and add analysis-only engineered columns."""
    cleaned = df.copy()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned.loc[cleaned["Expected_Ride_Duration"] <= 0, "Expected_Ride_Duration"] = np.nan
    cleaned.loc[cleaned["Number_of_Riders"] < 0, "Number_of_Riders"] = np.nan
    cleaned.loc[cleaned["Number_of_Drivers"] < 0, "Number_of_Drivers"] = np.nan
    cleaned.loc[~cleaned["Average_Ratings"].between(0, 5), "Average_Ratings"] = np.nan
    cleaned.loc[cleaned["Number_of_Past_Rides"] == 0, "Average_Ratings"] = cleaned["Average_Ratings"].mean()

    cleaned = cleaned.dropna(subset=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)

    # Cost_per_Minute contains TARGET, so it is EDA-only and never a model feature.
    cleaned["Cost_per_Minute"] = cleaned[TARGET] / cleaned["Expected_Ride_Duration"]
    cleaned["Supply_and_Demand"] = cleaned["Number_of_Riders"] / cleaned["Number_of_Drivers"].replace(0, 1)

    return cleaned

def make_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode prediction-ready features without target leakage."""
    cleaned = clean_data(df)
    encoded = pd.get_dummies(cleaned, columns=CATEGORICAL_COLUMNS, dtype=int)
    X = encoded.drop(columns=[TARGET, "Cost_per_Minute"])
    return X, encoded[TARGET]


def _models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, min_samples_leaf=5),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3, n_jobs=-1),
    }

def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compare candidate regressors with identical five-fold CV splits."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in _models().items():
        scores = cross_validate(model, X, y, cv=cv, n_jobs=-1, scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        })
        rows.append({
            "Model": name,
            "CV MAE ($)": -scores["test_mae"].mean(),
            "CV RMSE ($)": -scores["test_rmse"].mean(),
            "CV R2": scores["test_r2"].mean()
        })
    return pd.DataFrame(rows).sort_values("CV RMSE ($)", ignore_index=True)

def train_best_model(df: pd.DataFrame) -> dict:
    """Select by CV RMSE, then evaluate on a separate held-out test set."""
    X, y = make_training_data(df)
    comparison = compare_models(X, y)
    best_name = comparison.loc[0, "Model"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = _models()[best_name]
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "model": model,
        "model_name": best_name,
        "comparison": comparison,
        "feature_names": X.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": {
            "r2": r2_score(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions)))
        }
    }