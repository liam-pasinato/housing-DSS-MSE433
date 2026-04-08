import joblib
import pandas as pd
import numpy as np
import json
import shap

# =========================
# LOAD MODEL ARTIFACTS
# =========================
artifacts = joblib.load("ML-model/best_model.pkl")

model = artifacts["model"]
rmse = artifacts["rmse"]
feature_order = artifacts["features"]
encoder = artifacts["encoder"]
neighborhood_map = artifacts["neighborhood_map"]

# =========================
# COLUMN MAPPING (UI → MODEL)
# =========================
COLUMN_MAPPING = {
    "SquareFeet": "GrLivArea",
    "Bedrooms": "BedroomAbvGr",
    "Bathrooms": "FullBath",
    "GarageSpots": "GarageCars",
    "GarageArea": "GarageArea",
    "YearBuilt": "YearBuilt",
    "OverallQual": "OverallQual",
    "Neighborhood": "Neighborhood",
    "TotalBsmtSF": "TotalBsmtSF",
    "BsmtFullBath": "BsmtFullBath",
    "BsmtHalfBath": "BsmtHalfBath",
    "HalfBath": "HalfBath"
}

# =========================
# MAIN PREDICTION FUNCTION
# =========================
def make_prediction(data: str, input_type="json") -> list:
    assert input_type == "json"

    records = json.loads(data)
    input_df = pd.DataFrame.from_records(records)

    # -------------------------
    # 1. Rename columns
    # -------------------------
    input_df = input_df.rename(columns=COLUMN_MAPPING)

    # -------------------------
    # 2. FILL MISSING FEATURES (SAFE DEFAULTS)
    # -------------------------
    defaults = {
        "TotalBsmtSF": 0,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "HalfBath": 0
    }

    for col, val in defaults.items():
        if col not in input_df.columns:
            input_df[col] = val

    # -------------------------
    # 3. FEATURE ENGINEERING (MATCH TRAINING EXACTLY)
    # -------------------------

    # Total Bathrooms
    input_df["TotalBathrooms"] = (
        input_df["FullBath"]
        + 0.5 * input_df["HalfBath"]
        + input_df["BsmtFullBath"]
        + 0.5 * input_df["BsmtHalfBath"]
    )

    # Total SqFt
    input_df["TotalSqFt"] = input_df["GrLivArea"] + input_df["TotalBsmtSF"]

    # Age
    input_df["Age"] = 2025 - input_df["YearBuilt"]
    input_df["AgeSquared"] = input_df["Age"] ** 2

    # Interactions
    input_df["TotalRooms"] = input_df["BedroomAbvGr"] + input_df["FullBath"]
    input_df["BathPerBedroom"] = input_df["TotalBathrooms"] / (input_df["BedroomAbvGr"] + 1)
    input_df["SqFtPerRoom"] = input_df["TotalSqFt"] / (input_df["TotalRooms"] + 1)
    input_df["QualityAdjustedSqFt"] = input_df["OverallQual"] * input_df["GrLivArea"]

    # Neighborhood mean (FIXED)
    input_df["NeighborhoodPriceMean"] = input_df["Neighborhood"].map(neighborhood_map)
    input_df["NeighborhoodPriceMean"].fillna(
        np.mean(list(neighborhood_map.values())),
        inplace=True
    )

    # -------------------------
    # 4. ENCODE NEIGHBORHOOD
    # -------------------------
    encoded = encoder.transform(input_df[["Neighborhood"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["Neighborhood"])
    )

    input_df = input_df.drop(columns=["Neighborhood"]).reset_index(drop=True)
    input_df = pd.concat([input_df, encoded_df], axis=1)

    # -------------------------
    # 5. ALIGN FEATURES
    # -------------------------
    input_df = input_df.reindex(columns=feature_order, fill_value=0)

    # -------------------------
    # 6. PREDICT (LOG → NORMAL)
    # -------------------------
    log_prediction = model.predict(input_df)
    prediction = np.expm1(log_prediction)

    # -------------------------
    # 7. CONFIDENCE INTERVAL
    # -------------------------
    ci_lower = prediction - 1.96 * rmse
    ci_upper = prediction + 1.96 * rmse

    # -------------------------
    # 8. SHAP EXPLANATION
    # -------------------------
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_row = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

        explanation = {
            feature: float(val)
            for feature, val in zip(input_df.columns, shap_row)
        }

    except Exception as e:
        print("SHAP error:", e)
        explanation = {}

    # -------------------------
    # 9. RETURN
    # -------------------------
    return [{
        "prediction": float(prediction[0]),
        "ci_lower": float(ci_lower[0]),
        "ci_upper": float(ci_upper[0]),
        "explanation": explanation,
        "shap_values": explanation,
        "features_input": input_df.iloc[0].to_dict()
    }]