Project Overview

This project predicts the number of units sold for a product in a store using a Random Forest Regression model. It consists of two parts: model training and deployment via Streamlit. Users can interactively select features such as store ID, SKU ID, pricing, product display/feature status, and date details to get real-time predictions.

Features

Data Preprocessing: Handles missing values and removes outliers.

Feature Engineering: Extracts day, month, and year from the week column.

Model Training: Random Forest Regressor trained on historical sales data.

Model Deployment: Pre-trained model loaded in a Streamlit app for real-time predictions.

Interactive Interface: Dropdowns populated dynamically from dataset values.

Technologies Used

Python (pandas, scikit-learn, pickle)

Streamlit for web interface

Machine Learning (Random Forest Regressor)
