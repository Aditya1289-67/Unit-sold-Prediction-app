import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load your dataset (used only to populate field options)
df = pd.read_csv("myfile.csv")

# read training data
df2 = pd.read_csv("train_0irEZ2H.csv")

#check if total price has null in it
df2 = df2.drop(df2[df2['total_price'].isnull()].index)

#converting week column into datetime
df2['week'] = pd.to_datetime(df2['week'], format='%d/%m/%y')

#converting week column into day month and year
df2['day'] = df2['week'].dt.day
df2['month'] = df2['week'].dt.month
df2['year'] = df2['week'].dt.year
print(df2)

#only selecting data that is slightly normalised ( basically removing some outliers )
df21=df2[df2['units_sold']<df2['units_sold'].quantile(0.98)]

from sklearn.model_selection import train_test_split
#specifying target and Train features
# Features (X) and Target (y)
X = df21.drop(['units_sold','record_ID','week'], axis=1)  # Replace 'target' with your actual target column name
y = df21['units_sold']

# Split data into 80% train and 20% test
#splitting data into train and test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Initialize and train the Random Forest model
print(X_train.columns.to_list())
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_model.score(X_train,y_train)
# Save the model to a file

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
# Load the model from file
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

