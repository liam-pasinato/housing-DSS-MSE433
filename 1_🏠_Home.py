import json
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.datasets import fetch_openml

import DR_Predict
import Helpers

# Webpage title & icon 
Helpers.title_and_logo()

# Hide streamlit menu & watermark
Helpers.hide_streamlit_menu()

# Title
st.title("Ames, Iowa Real Estate Market")

@st.cache_data
def load_ames_data():
    raw_data = fetch_openml(name="house_prices", as_frame=True)
    return raw_data.frame

ames_df = load_ames_data()

# "About" dropdown
with st.expander("About:"):
    st.write(
        "This interactive Decision Support System (DSS) uses a machine learning model trained on the Ames Housing Dataset to predict residential property values and support renovation decisions."
    )
    st.write(
        "The model is built and deployed locally using scikit-learn, incorporating advanced features such as property quality, neighborhood effects, and engineered attributes (e.g., total square footage and bathroom ratios). It generates price predictions along with confidence intervals and model-driven explanations using SHAP values to highlight the key factors influencing each estimate."
    )
    st.write(
        "On the “House Viewings” page, you can explore filtered listings, compare home characteristics, and better understand how features like size, quality, and location impact price."
    )
    st.write(
        "On the “Price Prediction” page, you can input custom property details to receive a predicted value, confidence interval, and feature impact breakdown. The app also includes a renovation analysis tool that evaluates potential upgrades (e.g., adding bedrooms, bathrooms, or square footage), calculates return on investment (ROI), and identifies the maximum renovation budget for profitability. Additionally, the system provides ranked, ROI-based improvement suggestions to guide optimal investment decisions."
    )

st.header("🏘️ Neighborhood Price per Square Foot")
agg_df = Helpers.get_price_per_sqft_by_neighborhood(ames_df)
Helpers.plot_price_heatmap(agg_df)
