from collections import defaultdict

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


@st.cache(allow_output_mutation=True)
def read_data(path):
    try:
        return st.session_state["base_data"]
    except KeyError:
        df = pd.read_csv(path).loc[lambda x: ~pd.isna(x.bedrooms)]
        st.session_state["base_data"] = df
        return df

def plot_choropleth(geo_df, geo_json, color_column):
    fig = px.choropleth_mapbox(
        geo_df,
        geojson=geo_json,
        color=color_column,
        locations="Map ID",
        featureidkey="properties.id_col",
        center={"lat": 40.1608, "lon": -110.8910},
        mapbox_style="carto-positron",
        zoom=4.7,
        labels={"price_per_sq_ft": "Price per Sq. Foot"},
    )

    fig.update_layout(
        font=dict(size=16, color="Black"),
        title={
            "text": "Displaying Data for Featured Zipcodes",
            "x": 0.4,
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    return st.plotly_chart(fig)


def prediction_variables(df, zip_df):
    explanation_dict = {}

    for i in range(1, 4):
        explanation_dict[i - 1] = {}
        explanation_dict[i - 1]["expl"] = clean_string(
            df[f"EXPLANATION_{i}_FEATURE_NAME"].iloc[0]
        )
        explanation_dict[i - 1]["value"] = df[f"EXPLANATION_{i}_ACTUAL_VALUE"].iloc[0]
        explanation_dict[i - 1]["str"] = (
            df[f"EXPLANATION_{i}_STRENGTH"].iloc[0]
        ).round(3)
        explanation_dict[i - 1]["qual_str"] = df[
            f"EXPLANATION_{i}_QUALITATIVE_STRENGTH"
        ].iloc[0]
        if explanation_dict[i - 1]["expl"] == "Zip Geometry ":

            geometry = str(df["zip_geometry"].iloc[0])

            map_id = list(
                zip_df.loc[geometry == zip_df["zip_geometry"].astype(str), "Map ID"]
            )[0]
            explanation_dict[i - 1]["expl"] = f"Map ID #{map_id} "
            pp_sqft = list(
                zip_df.loc[
                    geometry == zip_df["zip_geometry"].astype(str), "price_per_sq_ft"
                ]
            )[0]
            explanation_dict[i - 1]["value"] = f"Price/Sq. Ft - ${pp_sqft}"

    return explanation_dict


def clean_columns(df, column_name):
    df = df.copy()

    df[column_name] = df[column_name].fillna(-1)
    df[column_name] = df[column_name].astype(int)
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].replace("-1", np.nan)

    return df[column_name]


def clean_string(explanation):
    string_array = explanation.split("_")
    temp_string = ""

    for count, i in enumerate(string_array):
        string_array[count] = i.capitalize()
        temp_string = temp_string + string_array[count] + " "

    clean_str = temp_string
    return clean_str


def filter_range(df, range, column_name):
    if range == "ANY":
        return df

    range_arr = range.split(" - ")
    low_range = int(range_arr[0])
    high_range = int(range_arr[1])
    range_check = (df[column_name] >= low_range) & (df[column_name] <= high_range)

    return df.loc[range_check]


def get_explanatory_data(json_response, zip):
    explanations = defaultdict(list)
    pred_explanations = json_response[0]["predictionExplanations"]

    for i in range(3):
        feature = pred_explanations[i]["feature"]
        value = pred_explanations[i]["featureValue"]
        strength = round(pred_explanations[i]["strength"], 3)

        if feature == "zip_geometry":
            geo = value
            map_id = list(zip.loc[geo == zip["zip_geometry"].astype(str), "Map ID"])[0]

            feature = f"Map ID #{map_id} "

            price_sqft = list(
                zip.loc[geo == zip["zip_geometry"].astype(str), "price_per_sq_ft"]
            )[0]

            value = f"Price/Sq. Ft - ${price_sqft}"

        else:
            feature = clean_string(feature)

        explanations["feature"].append(feature)
        explanations["value"].append(value)
        explanations["strength"].append(strength)
        explanations["qual_strength"].append(
            pred_explanations[i]["qualitativeStrength"]
        )
    return explanations


def write_explanations(container, container_title, exp_dict):
    container_title.write("> ### Prediction Explanations")

    with container:
        st.write(
            "#### Showing the 3 features with the largest impact on your prediction:"
        )
        exp1, exp2, exp3 = st.columns(3, gap="medium")

        with exp1:
            feat1 = st.expander(
                f"1. {exp_dict['feature'][0]} {exp_dict['qual_strength'][0]}"
            )
            feat1.write(f"{exp_dict['feature'][0]}: {exp_dict['value'][0]}")
            feat1.write(f"Strength: {exp_dict['strength'][0]}")

            if exp_dict["strength"][0] >= 0:
                feat1.write(f"{exp_dict['feature'][0]} positively impacts prediction")
            else:
                feat1.write(f"{exp_dict['feature'][0]} negatively impacts prediction")

        with exp2:
            feat2 = st.expander(
                f"2. {exp_dict['feature'][1]} {exp_dict['qual_strength'][1]}"
            )
            feat2.write(f"{exp_dict['feature'][1]}: {exp_dict['value'][1]}")
            feat2.write(f"Strength: {exp_dict['strength'][1]}")

            if exp_dict["strength"][1] >= 0:
                feat2.write(f"{exp_dict['feature'][1]} positively impacts prediction")
            else:
                feat2.write(f"{exp_dict['feature'][1]} negatively impacts prediction")

        with exp3:
            feat3 = st.expander(
                f"3. {exp_dict['feature'][2]} {exp_dict['qual_strength'][2]}"
            )
            feat3.write(f"{exp_dict['feature'][2]}: {exp_dict['value'][2]}")
            feat3.write(f"Strength: {exp_dict['strength'][2]}")

            if exp_dict["strength"][2] >= 0:
                feat3.write(f"{exp_dict['feature'][2]} positively impacts prediction")
            else:
                feat3.write(f"{exp_dict['feature'][2]} negatively impacts prediction")
        return


def make_plot(df, xaxis_choice):
    fig = px.scatter(
        df,
        x=xaxis_choice,
        y="price_PREDICTION",
        labels={
            "price_PREDICTION": "Predicted Price",
            "sq_ft": "Square Footage",
            "price": "Actual Price",
            "acres": "Acres",
            "bathrooms": "Bathrooms",
            "bedrooms": "Bedrooms",
            "is_New": "Predicted Home",
        },
        color="is_New",
        symbol="is_New",
    )


def write_view_explanation(real_price, est_price, img, listing, dict):
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.write("> #### Real Price: " + "${:,.2f}".format(real_price))
        st.image(img, caption=f"Listing ID: {listing}")

    with col2:
        st.write("> #### Predicted Price: " + "${:,.2f}".format(est_price))
        st.write("##### Factors influencing house price:")

        # Explanation 1
        feat1 = st.expander(f'1. {dict[0]["expl"]} {dict[0]["qual_str"]}')
        feat1.write(f'{dict[0]["expl"]}: {dict[0]["value"]}')
        feat1.write(f'Strength: {dict[0]["str"]}')
        if dict[0]["str"] >= 0:
            feat1.write(f'{dict[0]["expl"]} positively impacts prediction')
        else:
            feat1.write(f'{dict[0]["expl"]} negatively impacts prediction')

        # Explanation 2
        feat2 = st.expander(f'2. {dict[1]["expl"]} {dict[1]["qual_str"]}')
        feat2.write(f'{dict[1]["expl"]}: {dict[1]["value"]}')
        feat2.write(f'Strength: {dict[1]["str"]}')
        if dict[1]["str"] >= 0:
            feat2.write(f'{dict[1]["expl"]} positively impacts prediction')
        else:
            feat2.write(f'{dict[1]["expl"]} negatively impacts prediction')

        # Explanation 3
        feat3 = st.expander(f'3. {dict[2]["expl"]} {dict[2]["qual_str"]}')
        feat3.write(f'{dict[2]["expl"]}: {dict[2]["value"]}')
        feat3.write(f'Strength: {dict[2]["str"]}')
        if dict[2]["str"] >= 0:
            feat3.write(f'{dict[2]["expl"]} positively impacts prediction')
        else:
            feat3.write(f'{dict[2]["expl"]} negatively impacts prediction')

    return


def title_and_logo():
    logo = Image.open("./Data/DR_logo.png")
    st.set_page_config(page_title="Utah Housing Market", page_icon=logo)
    return


def hide_streamlit_menu():
    hide_menu = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu, unsafe_allow_html=True)
    return

neighborhood_coords = {
"NAmes": (42.0559, -93.6395), "CollgCr": (42.0266, -93.6481), "OldTown": (42.0347, -93.6196),
"Edwards": (42.0179, -93.6151), "Somerst": (42.0303, -93.6632), "Gilbert": (42.1031, -93.6434),
"NridgHt": (42.0701, -93.6505), "Sawyer": (42.0362, -93.6589), "Mitchel": (42.0003, -93.6512),
"SawyerW": (42.0401, -93.6702), "BrkSide": (42.0212, -93.6278), "ClearCr": (42.0782, -93.6418),
"Crawfor": (42.0174, -93.6249), "NoRidge": (42.0742, -93.6576), "Timber": (42.0781, -93.6762),
"IDOTRR": (42.0182, -93.6110), "NPkVill": (42.0319, -93.6101), "Blmngtn": (42.0170, -93.6343),
"NWAmes": (42.0718, -93.6625), "StoneBr": (42.0824, -93.6571), "MeadowV": (42.0036, -93.6235),
"SWISU": (42.0131, -93.6192), "Blueste": (42.0009, -93.6067), "Greens": (42.0841, -93.6659),
"GrnHill": (42.0952, -93.6730), "Veenker": (42.0330, -93.6590)
}

def get_price_per_sqft_by_neighborhood(df):
    df = df.copy()
    df = df[df["GrLivArea"] > 0] # prevent division by zero
    df["price_per_sqft"] = df["SalePrice"] / df["GrLivArea"]
    return df.groupby("Neighborhood")["price_per_sqft"].mean().reset_index()

def plot_price_heatmap(df):
    df["lat"] = df["Neighborhood"].map(lambda x: neighborhood_coords.get(x, (None, None))[0])
    df["lon"] = df["Neighborhood"].map(lambda x: neighborhood_coords.get(x, (None, None))[1])
    
    fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="price_per_sqft",
    size="price_per_sqft",
    hover_name="Neighborhood",
    color_continuous_scale="YlOrRd",
    size_max=20,
    zoom=11,
    mapbox_style="carto-positron",
    )

    fig.update_layout(
    title="Average Price per Square Foot by Neighborhood",
    font=dict(size=14),
    title_x=0
    )
    return st.plotly_chart(fig, use_container_width=True)