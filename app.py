import os
import streamlit as st
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Models
kmeans = joblib.load(
    os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)

columns = joblib.load(
    os.path.join(BASE_DIR, "models", "columns.pkl")
)

country_encoder = joblib.load(
    os.path.join(BASE_DIR, "models", "country_encoder.pkl")
)

st.title("🛒 Customer Segmentation App")

st.write("Enter customer details to identify the customer segment.")

# ==========================
# Inputs
# ==========================

quantity = st.number_input(
    "Quantity Purchased",
    min_value=1,
    value=5
)

unit_price = st.number_input(
    "Unit Price",
    min_value=0.0,
    value=10.0
)

country = st.selectbox(
    "Country",
    list(country_encoder.classes_)
)

# ==========================
# Feature Engineering
# ==========================

revenue = quantity * unit_price

country_label = country_encoder.transform(
    [country]
)[0]

# ==========================
# Create Input Vector
# ==========================

input_data = np.zeros(len(columns))

input_dict = dict(
    zip(columns, input_data)
)

if "Quantity" in input_dict:
    input_dict["Quantity"] = quantity

if "UnitPrice" in input_dict:
    input_dict["UnitPrice"] = unit_price

if "Revenue" in input_dict:
    input_dict["Revenue"] = revenue

if "Country_Label" in input_dict:
    input_dict["Country_Label"] = country_label

final_input = np.array(
    list(input_dict.values())
).reshape(1, -1)

# ==========================
# Scaling
# ==========================

final_scaled = scaler.transform(
    final_input
)

# ==========================
# Predict Cluster
# ==========================

if st.button("Predict Segment"):

    cluster = kmeans.predict(
        final_scaled
    )[0]

    segment_names = {
        0: "High Value Customers",
        1: "Regular Customers",
        2: "Occasional Customers",
        3: "Low Value Customers"
    }

    segment = segment_names.get(
        cluster,
        f"Cluster {cluster}"
    )

    st.success(
        f"Predicted Customer Segment: {segment}"
    )

    st.write(f"Cluster ID: {cluster}")

    st.info(
        f"""
        Quantity : {quantity}

        Unit Price : ₹ {unit_price:.2f}

        Revenue : ₹ {revenue:.2f}

        Country : {country}
        """
    )