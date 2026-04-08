# Housing Decision Support System (DSS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

## Overview

This project is an **end-to-end machine learning Decision Support System (DSS)** that predicts home prices and provides **data-driven renovation recommendations**.

Built using the **Ames Housing Dataset**, the system goes beyond prediction by offering:
-  **Price estimates with confidence intervals**
-  **SHAP-based feature explanations**
-  **Renovation ROI analysis**
-  **Ranked improvement suggestions**

## 🎥 Demo

https://github.com/liam-pasinato/housing-DSS-MSE433/433_demo.mov

---

## ✨ Key Features

- **Accurate Price Prediction**
  - HistGradientBoosting model (RMSE ≈ **$22K**)
  - Log-transformed target + cross-validation

- **Explainability (SHAP)**
  - Visual feature impact chart
  - Top drivers of price highlighted with insights

- **Renovation Analyzer**
  - Simulate adding:
    - Bedrooms / Bathrooms (supports half-baths)
    - Square footage
  - ROI indicators (🟢 profitable / 🔴 not profitable)
  - Ranked **Top 3 improvements**

- **Interactive UI**
  - Built with Streamlit
  - Multi-page app with filtering & exploration

---

## Machine Learning Highlights

- **Models Compared:**
  - Random Forest
  - Gradient Boosting
  - ✅ HistGradientBoosting (Best)

- **Final Model Performance:**
  - RMSE: **$22,256**
  - 3-fold Cross Validation (log space)
  - Monotonic constraint on square footage

- **Feature Engineering:**
  - Interaction features (quality × size)
  - Neighborhood price mean
  - Bath/bed & sqft/room ratios
  - Basement + total feature aggregation

---

## 🛠️ Tech Stack

- **Python**
- **scikit-learn**
- **SHAP**
- **Streamlit**
- **Pandas / NumPy**

---

## ⚙️ How to Run

```bash
git clone https://github.com/YOUR_USERNAME/housing-DSS-MSE433.git
cd housing-DSS-MSE433

pip install -r requirements.txt

streamlit run app/1_🏠_Home.py
```

---

## Example Capabilities:
  - Predict home price from custom input
  - Understand why a home is valued a certain way
  - Compare current vs renovated home value
  - Identify highest ROI improvements

## Impact
This system transforms a standard ML model into a decision-making tool, enabling:

  - Smarter home pricing
  - Data-driven renovation planning
  - Transparent and interpretable predictions

## Course Integration:
  - Statistics (MSE 251/253): Confidence intervals & uncertainty
  - Machine Learning (MSE 446/546): Model training, tuning, evaluation
  - Decision Support Systems (MSE 436): Interactive dashboards & ROI tools
  - HCI (MSE 343): User-centered design & visualization

## Future Improvements:
  - XGBoost / LightGBM exploration
  - More granular neighborhood features
  - Real-time data integration
  - Advanced renovation scenarios (kitchen, exterior, etc.)
