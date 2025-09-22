import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load your dataset (used only to populate field options)
df = pd.read_csv(r"C:\Users\HP\Desktop\Unit sold Prediction\myfile.csv ")
df['week'] = pd.to_datetime(df['week'], format='%d/%m/%y')
df['day'] = df['week'].dt.day
df['month'] = df['week'].dt.month
df['year'] = df['week'].dt.year
df = df.drop(columns=['week'])

# Drop unnecessary columns
df = df.drop(columns=['record_ID', 'week'], errors='ignore')

# Load your trained model
with open(r"C:\Users\HP\Desktop\Unit sold Prediction\random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Unit Sold Prediction App Test 1")

st.subheader("Select Inputs (values taken from dataset)")

# Use dropdowns populated from the CSV dataset
store_id = st.selectbox("Store ID", sorted(df['store_id'].unique()))
sku_id = st.selectbox("SKU ID", sorted(df['sku_id'].unique()))
total_price = st.selectbox("Total Price", sorted(df['total_price'].unique()))
base_price = st.selectbox("Base Price", sorted(df['base_price'].unique()))
is_featured_sku = st.selectbox("Is Featured SKU?", df['is_featured_sku'].unique())
is_display_sku = st.selectbox("Is Displayed SKU?", df['is_display_sku'].unique())
day = st.selectbox("Day", sorted(df['day'].unique()))
month = st.selectbox("Month", sorted(df['month'].unique()))
year = st.selectbox("Year", sorted(df['year'].unique()))

# Prediction trigger
if st.button("Predict Unit Sold"):
    input_data = pd.DataFrame({
        "store_id": [store_id],
        "sku_id": [sku_id],
        "total_price": [total_price],
        "base_price": [base_price],
        "is_featured_sku": [is_featured_sku],
        "is_display_sku": [is_display_sku],
        "day": [day],
        "month": [month],
        "year": [year]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"🔢 Predicted Units Sold: {round(prediction, 2)}")