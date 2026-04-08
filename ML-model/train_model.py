import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# 1. Load dataset
ames = fetch_openml(name="house_prices", as_frame=True)
df = ames.frame


# =========================
# 2. SELECT FEATURES
# =========================
features = [
    "GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath",
    "GarageCars", "GarageArea",
    "YearBuilt", "OverallQual",
    "Neighborhood",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"
]

target = "SalePrice"

df = df[features + [target]].dropna()

# =========================
# 3. FEATURE ENGINEERING
# =========================

# Total Bathrooms
df["TotalBathrooms"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)

# Total Square Footage
df["TotalSqFt"] = df["GrLivArea"] + df["TotalBsmtSF"]

# Age
df["Age"] = 2025 - df["YearBuilt"]
df["AgeSquared"] = df["Age"] ** 2

# Interactions
df["TotalRooms"] = df["BedroomAbvGr"] + df["FullBath"]
df["BathPerBedroom"] = df["TotalBathrooms"] / (df["BedroomAbvGr"] + 1)
df["SqFtPerRoom"] = df["TotalSqFt"] / (df["TotalRooms"] + 1)
df["QualityAdjustedSqFt"] = df["OverallQual"] * df["GrLivArea"]

# Neighborhood Mean Price
neighborhood_map = df.groupby("Neighborhood")[target].mean().to_dict()
df["NeighborhoodPriceMean"] = df["Neighborhood"].map(neighborhood_map)

# =========================
# 4. REMOVE OUTLIERS
# =========================
df = df[df["SalePrice"] < df["SalePrice"].quantile(0.99)]
df = df[df["GrLivArea"] < df["GrLivArea"].quantile(0.99)]

# =========================
# 5. LOG TRANSFORM TARGET
# =========================
y = np.log1p(df[target])

# =========================
# 6. ENCODE CATEGORICAL
# =========================
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoded = encoder.fit_transform(df[["Neighborhood"]])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["Neighborhood"])
)

# Drop original categorical
df = df.drop(columns=["Neighborhood"]).reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

# =========================
# 7. FINAL FEATURE SET
# =========================
X = df.drop(columns=[target])
feature_order = X.columns.tolist()

# =========================
# 8. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 9. BASELINE MODEL (RF)
# =========================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_preds = np.expm1(rf.predict(X_test))
rf_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), rf_preds))

print("\n📊 Random Forest Results:")
print(f"   RMSE: {rf_rmse:,.2f}")
print(f"   Mean Predicted Price: ${rf_preds.mean():,.0f}")
print(f"   Mean Actual Price:    ${np.expm1(y_test).mean():,.0f}")

# =========================
# 9B. GRADIENT BOOSTING MODEL
# =========================
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gbr.fit(X_train, y_train)

# Predictions (convert back from log)
gbr_preds = np.expm1(gbr.predict(X_test))

# RMSE in original price space
gbr_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), gbr_preds))

print("\n📊 Gradient Boosting Results:")
print(f"   RMSE: {gbr_rmse:,.2f}")
print(f"   Mean Predicted Price: ${gbr_preds.mean():,.0f}")
print(f"   Mean Actual Price:    ${np.expm1(y_test).mean():,.0f}")

# =========================
# 10. HIST GRADIENT BOOSTING (MONOTONIC)
# =========================

# Monotonic constraints (only for GrLivArea)
monotonic_constraints = [0] * X_train.shape[1]

if "GrLivArea" in feature_order:
    monotonic_constraints[feature_order.index("GrLivArea")] = 1

hgb = HistGradientBoostingRegressor(
    random_state=42,
    monotonic_cst=monotonic_constraints
)

# =========================
# 11. LIGHT HYPERPARAMETER TUNING
# =========================
param_dist = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [150, 200, 300]
}

search = RandomizedSearchCV(
    hgb,
    param_distributions=param_dist,
    n_iter=8,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_hgb = search.best_estimator_

# =========================
# 12. CROSS-VALIDATION SCORE
# =========================
cv_rmse = -cross_val_score(
    best_hgb,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=3
).mean()

# =========================
# 13. FINAL EVALUATION
# =========================
hgb_preds = np.expm1(best_hgb.predict(X_test))
hgb_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), hgb_preds))

print("\n📊 HistGradientBoosting (Tuned) Results:")
print(f"   RMSE: {hgb_rmse:,.2f}")
print(f"   CV RMSE (log space): {cv_rmse:.4f}")
print(f"   Mean Predicted Price: ${hgb_preds.mean():,.0f}")
print(f"   Mean Actual Price:    ${np.expm1(y_test).mean():,.0f}")

# =========================
# 14. SELECT BEST MODEL
# =========================
# FINAL MODEL EVALUATION (Comparison)
print("\n🏆 MODEL COMPARISON SUMMARY")
print("-" * 40)
print(f"Random Forest RMSE:           {rf_rmse:,.2f}")
print(f"Gradient Boosting RMSE:       {gbr_rmse:,.2f}")
print(f"HistGradientBoosting RMSE:    {hgb_rmse:,.2f}")
print("-" * 40)

model_results = {
    "RandomForest": (rf, rf_rmse),
    "GradientBoosting": (gbr, gbr_rmse),
    "HistGradientBoosting": (best_hgb, hgb_rmse)
}

best_model_name = min(model_results, key=lambda k: model_results[k][1])
best_model, best_rmse = model_results[best_model_name]

print(f"\n✅ Best Model Selected: {best_model_name}")
print(f"   Final RMSE: {best_rmse:,.2f}")

# =========================
# 15. SAVE ARTIFACTS
# =========================
artifacts = {
    "model": best_model,
    "rmse": best_rmse,
    "features": feature_order,
    "encoder": encoder,
    "neighborhood_map": neighborhood_map
}

joblib.dump(artifacts, "best_model.pkl")

print("✅ Model saved to best_model.pkl")