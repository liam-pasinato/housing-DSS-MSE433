import streamlit as st
import pandas as pd
import numpy as np
import DR_Predict

# =========================
# PAGE SETUP
# =========================
st.title("Home Appraiser & Renovation Analyzer")

# -------------------------
# 1️. Initialize session state for inputs
# -------------------------
for key, default in {
    "SquareFeet": 1500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "GarageSpots": 1,
    "GarageArea": 200,
    "YearBuilt": 2000,
    "OverallQual": 5,
    "Neighborhood": "NAmes",
    "renovation_type": "Add Bdrm/Bath",
    "bedrooms_to_add": 0,
    "bathrooms_to_add": 0,
    "sqft_to_add": 0,
    "renovation_cost": 0,
    "last_prediction": None,
    "last_ci_lower": None,
    "last_ci_upper": None,
    "last_explanation": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# 2️. Input form for house details
# -------------------------
with st.form("house_input_form"):
    st.header("🏠 Enter Home Details")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.SquareFeet = st.number_input("Square Feet", min_value=100, max_value=10000,
                                                      value=st.session_state.SquareFeet, step = 100)
        st.session_state.Bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10,
                                                    value=st.session_state.Bedrooms)
        st.session_state.Bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10,
                                                     value=st.session_state.Bathrooms)
        st.session_state.GarageSpots = st.number_input("Garage Spots", min_value=0, max_value=10,
                                                       value=st.session_state.GarageSpots)
    with col2:
        st.session_state.GarageArea = st.number_input("Garage Area (sqft)", min_value=0, max_value=2000,
                                                      value=st.session_state.GarageArea, step = 50)
        st.session_state.YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025,
                                                     value=st.session_state.YearBuilt)
        st.session_state.OverallQual = st.slider("Overall Quality", min_value=1, max_value=10,
                                                 value=st.session_state.OverallQual)
        st.session_state.Neighborhood = st.selectbox(
            "Neighborhood",
            options=[
                "NAmes", "CollgCr", "OldTown", "Edwards",
                "Somerst", "Gilbert", "NridgHt", "Sawyer"]
        )

    submit = st.form_submit_button("Predict Price")

# -------------------------
# 3️. Make prediction when form submitted
# -------------------------
if submit:
    input_data = pd.DataFrame([{
        "SquareFeet": st.session_state.SquareFeet,
        "Bedrooms": st.session_state.Bedrooms,
        "Bathrooms": st.session_state.Bathrooms,
        "GarageSpots": st.session_state.GarageSpots,
        "GarageArea": st.session_state.GarageArea,
        "YearBuilt": st.session_state.YearBuilt,
        "OverallQual": st.session_state.OverallQual,
        "Neighborhood": st.session_state.Neighborhood
    }])

    res = DR_Predict.make_prediction(input_data.to_json(orient="records"))

    st.session_state.last_prediction = res[0]["prediction"]
    st.session_state.last_ci_lower = res[0]["ci_lower"]
    st.session_state.last_ci_upper = res[0]["ci_upper"]
    st.session_state.last_explanation = res[0]["explanation"]
    st.session_state.last_shap_values = res[0]["shap_values"]

    st.session_state.res = res
    st.session_state.features_input = res[0]["features_input"]
    st.session_state.raw_input = input_data.iloc[0].to_dict()

# -------------------------
# 4️. Display prediction & CI
# -------------------------
if st.session_state.last_prediction is not None:
    st.subheader("Predicted Home Price")
    st.subheader(f"${st.session_state.last_prediction:,.0f}")
    st.write(f"95% Confidence Interval: ${st.session_state.last_ci_lower:,.0f} - ${st.session_state.last_ci_upper:,.0f}")
    st.divider()

    # -------------------------
    # 5️. SHAP Feature Impact
    # -------------------------
    top_shap = dict(
        sorted(st.session_state.last_shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    st.subheader("Feature Impact on Predicted Price")
    if top_shap:
        shap_df = pd.DataFrame({
            "Feature": list(top_shap.keys()),
            "Impact": list(top_shap.values())
        }).sort_values("Impact", ascending=True)

        st.bar_chart(shap_df.set_index("Feature")["Impact"])
    else:
        st.write("SHAP explanation unavailable.")

    def interpret_feature(feature_name, shap_value, feature_value):
        direction = "up" if shap_value > 0 else "down"
        return f"The {feature_name} of this home ({feature_value}) is a key factor driving its value {direction}."

    # Key Insights
if "res" in st.session_state:
    shap_vals = st.session_state.res[0]["shap_values"]
    feature_vals = st.session_state.features_input

    top_3 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    st.write("### Top Feature Insights:")
    for feat, val in top_3:
        feature_val = feature_vals[feat]  # just the value for this home
        insight = interpret_feature(feat, val, feature_val)
        st.write(f"- {insight}")
    
    st.divider()

# -------------------------
# 6️. Renovation Analyzer
# -------------------------
st.subheader("🔨 Renovation Analyzer")

#if st.session_state.last_prediction is None:
if "res" not in st.session_state:
    st.info("Please simulate a house scenario to use the renovation analyzer.")
else:
    st.radio("Renovation Type", ["Add Bdrm/Bath", "Add SqFt"], key="renovation_type")

    with st.form("renovation_form"):
        if st.session_state.renovation_type == "Add Bdrm/Bath":
            st.session_state.bedrooms_to_add = st.number_input("Bedrooms to Add", min_value=0, max_value=5,
                                                               value=st.session_state.bedrooms_to_add)
            st.session_state.bathrooms_to_add = st.number_input("Bathrooms to Add", min_value=0, max_value=5,
                                                                value=st.session_state.bathrooms_to_add)
        else:
            st.session_state.sqft_to_add = st.number_input("Square Feet to Add", min_value=0, max_value=5000,
                                                           value=st.session_state.sqft_to_add, step = 100)

        st.session_state.renovation_cost = st.number_input("Estimated Renovation Cost ($)", min_value=0,
                                                           value=st.session_state.renovation_cost, step = 500)

        reno_submit = st.form_submit_button("Analyze Renovation")

    if reno_submit:
        # Compute new house configuration
        new_data = {
            "SquareFeet": st.session_state.SquareFeet + st.session_state.sqft_to_add,
            "Bedrooms": st.session_state.Bedrooms + st.session_state.bedrooms_to_add,
            "Bathrooms": st.session_state.Bathrooms + st.session_state.bathrooms_to_add,
            "GarageSpots": st.session_state.GarageSpots,
            "GarageArea": st.session_state.GarageArea,
            "YearBuilt": st.session_state.YearBuilt,
            "OverallQual": st.session_state.OverallQual,
            "Neighborhood": st.session_state.Neighborhood
        }
        new_input = pd.DataFrame([new_data])
        reno_res = DR_Predict.make_prediction(new_input.to_json(orient="records"))

        reno_cost = st.session_state.renovation_cost
        new_price = reno_res[0]["prediction"]
        old_price = st.session_state.last_prediction
        ci_lower = reno_res[0]["ci_lower"]
        ci_upper = reno_res[0]["ci_upper"]

        # --Output--
        st.write("## Renovation Analysis")

        # --- Core Calculations ---
        value_gain = new_price - old_price
        profit = value_gain - reno_cost
        roi = (profit / reno_cost) * 100 if reno_cost > 0 else 0
        break_even = value_gain  # max budget

        # --- Side-by-side metrics ---
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Current Value",
                value=f"${old_price:,.0f}"
            )

            # --- MAX BUDGET CARD ---
            st.write("### MAX Renovation Budget")

            if value_gain <= 0:
                st.info(f"Renovation does not seem to be valuable.")
            else:
                st.info(f"You should spend **no more than ${break_even:,.0f}** to break even.")

        with col2:
            st.metric(
                label="Renovated Value",
                value=f"${new_price:,.0f}",
                delta= round(value_gain)
            )

            # --- ROI Indicator ---
            st.write("### ROI Analysis")

            if profit > 0:
                st.success(f"Profitable: +${profit:,.0f} ({roi:.1f}% ROI)")
            elif profit < 0:
                st.error(f"Not Profitable: -${abs(profit):,.0f} ({roi:.1f}% ROI)")
            else:
                st.warning("Break-even renovation")

            # Optional stronger callout
            if reno_cost > break_even:
                st.warning("⚠️ Your current budget exceeds the break-even point.")
            else:
                st.success("✅ Your renovation is within a profitable budget.")

        st.divider()

        st.write("## Smart Renovation Suggestions (ROI-Based)")

        current_price = st.session_state.res[0]["prediction"]

        suggestions = []

        COSTS = {
            "Add 1 Bedroom": 9000,
            "Add 1 Bathroom": 6000,
            "Add 1 Bed 1 Bath": 13000,
            "Add 300 SqFt": 10000,
            "Add 600 SqFt": 18000,
            "Add 900 SqFt": 27000
        }

        # --- 1. Add Bedroom ---
        test = st.session_state.raw_input.copy()
        test["Bedrooms"] += 1

        new_price = DR_Predict.make_prediction(
            pd.DataFrame([test]).to_json(orient="records")
        )[0]["prediction"]

        gain = new_price - current_price
        cost = COSTS["Add 1 Bedroom"]
        profit = gain - cost
        roi = (profit / cost) * 100
        suggestions.append(("Add 1 Bedroom", gain, cost, profit, roi))

        # --- 2. Add Bathroom ---
        test = st.session_state.raw_input.copy()
        test["Bathrooms"] += 1

        new_price = DR_Predict.make_prediction(
            pd.DataFrame([test]).to_json(orient="records")
        )[0]["prediction"]

        gain = new_price - current_price
        cost = COSTS["Add 1 Bathroom"]
        profit = gain - cost
        roi = (profit / cost) * 100
        suggestions.append(("Add 1 Bathroom", gain, cost, profit, roi))

        # --- 3. Add 300 SqFt ---
        test = st.session_state.raw_input.copy()
        test["SquareFeet"] += 300

        new_price = DR_Predict.make_prediction(
            pd.DataFrame([test]).to_json(orient="records")
        )[0]["prediction"]

        gain = new_price - current_price
        cost = COSTS["Add 300 SqFt"]
        profit = gain - cost
        roi = (profit / cost) * 100
        suggestions.append(("Add 300 SqFt", gain, cost, profit, roi))

        # --- 4. Add 600 SqFt ---
        test = st.session_state.raw_input.copy()
        test["SquareFeet"] += 600

        new_price = DR_Predict.make_prediction(
            pd.DataFrame([test]).to_json(orient="records")
        )[0]["prediction"]

        gain = new_price - current_price
        cost = COSTS["Add 600 SqFt"]
        profit = gain - cost
        roi = (profit / cost) * 100
        suggestions.append(("Add 600 SqFt", gain, cost, profit, roi))

        # --- 5. Add 900 SqFt ---
        test = st.session_state.raw_input.copy()
        test["SquareFeet"] += 900

        new_price = DR_Predict.make_prediction(
            pd.DataFrame([test]).to_json(orient="records")
        )[0]["prediction"]

        gain = new_price - current_price
        cost = COSTS["Add 900 SqFt"]
        profit = gain - cost
        roi = (profit / cost) * 100
        suggestions.append(("Add 900 SqFt", gain, cost, profit, roi))

        # --- 6. Add Bed & Bath ---
        test = st.session_state.raw_input.copy()
        test["Bathrooms"] += 1
        test["Bedrooms"] += 1

        new_price = DR_Predict.make_prediction(
            pd.DataFrame([test]).to_json(orient="records")
        )[0]["prediction"]

        gain = new_price - current_price
        cost = COSTS["Add 1 Bed 1 Bath"]
        profit = gain - cost
        roi = (profit / cost) * 100
        suggestions.append(("Add 1 Bed & 1 Bath", gain, cost, profit, roi))

        # --- Sort Top 3 ---
        suggestions_sorted = sorted(suggestions, key=lambda x: x[3], reverse=True)

        profitable_suggestions = [s for s in suggestions_sorted if s[3] > 0]

        # --- Display ---
        if not suggestions_sorted:
            st.warning("No profitable renovations found for this home.")

        else:
            for i, (label, gain, cost, profit, roi) in enumerate(profitable_suggestions):
                
                if i == 0:
                    st.write("### 🏆 Best Investment Options")
                    st.write("Showcasing the most profitable projects from 6 simulated renovation scenarios.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"##### Renovation Idea")
                    st.write(f"**{label}**")

                with col2:
                    st.write(f"##### Added Home Value")
                    st.write(f"${gain:,.0f}")

                with col3:
                    st.write(f"##### Estimated ROI")
                    if profit > 0:
                        st.success(f"+${profit:,.0f} ({roi:.1f}%)")
                    else:
                        st.error(f"-${abs(profit):,.0f} ({roi:.1f}%)")
        