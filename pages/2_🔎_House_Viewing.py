from cgitb import small

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st 
from sklearn.datasets import fetch_openml

import DR_Predict
import Helpers

# Webpage title & icon
Helpers.title_and_logo()

# Hide streamlit menu & watermark
Helpers.hide_streamlit_menu()

@st.cache_data
def load_ames_data():
    data = fetch_openml(name="house_prices", as_frame=True)
    df = data.frame
    
        # Rename relevant columns for easier reference
    df = df.rename(columns={
        "GrLivArea": "Sq. Footage",
        "BedroomAbvGr": "Bed",
        "FullBath": "Bath",
        "GarageCars": "Car Garage",
        "SalePrice": "Price"
    })

    # Keep selected columns for filtering and display
    keep_cols = ["Sq. Footage", "Bed", "Bath", "Car Garage", "Price", "YearBuilt", "OverallQual", "TotRmsAbvGrd", "Neighborhood", "KitchenAbvGr", "HalfBath"]
    df = df[keep_cols].dropna()
    df[["Sq. Footage", "Bed", "Bath", "Car Garage", "Price"]] = df[["Sq. Footage", "Bed", "Bath", "Car Garage", "Price"]].astype(int)
    return df

df = load_ames_data()

st.write("# House Listings")
st.write("> ### Use filters below to find listings:")


col1, col2, col3 = st.columns(3)

with col1:
    bed_filt = st.multiselect('Filter by Bedrooms', sorted(df['Bed'].unique()))
    bath_filt = st.multiselect('Filter by Bathrooms', sorted(df['Bath'].unique()))

with col2:
    garage_filt = st.multiselect('Filter by Garage Size', sorted(df['Car Garage'].unique()))
    sqft_range = st.slider("Square Footage Range", min_value=int(df["Sq. Footage"].min()), max_value=int(df["Sq. Footage"].max()), value=(1000, 3000), step=100)

with col3:
    price_range = st.slider("Price Range", min_value=int(df["Price"].min()), max_value=int(df["Price"].max()), value=(100000, 400000), step=5000)

#Create Filtered Dataset based on imputs
filt_df = df.copy()

if bed_filt:
    filt_df = filt_df[filt_df["Bed"].isin(bed_filt)]

if bath_filt:
    filt_df = filt_df[filt_df["Bath"].isin(bath_filt)]

if garage_filt:
    filt_df = filt_df[filt_df["Car Garage"].isin(garage_filt)]

filt_df = filt_df[
    (filt_df["Sq. Footage"] >= sqft_range[0]) &
    (filt_df["Sq. Footage"] <= sqft_range[1]) &
    (filt_df["Price"] >= price_range[0]) &
    (filt_df["Price"] <= price_range[1])
]

st.write(f"### Showing {len(filt_df)} matching homes")
st.dataframe(filt_df.reset_index(drop=True), use_container_width=True)