This project predicts the number of units sold for products in stores using a Random Forest Regression model. The workflow uses a pre-split dataset (train, test, sample CSV) from Kaggle Demand Forecasting Dataset
.

The project has two main parts:

Model Training

Loads the train.csv for training the Random Forest model.

Performs data cleaning, removing nulls and outliers.

Feature engineering: converts the week column into day, month, and year.

Trains a Random Forest Regressor on historical sales data (units_sold target).

Saves the trained model as random_forest_model.pkl.

Streamlit Deployment

Loads the pre-trained model and allows users to input features through dropdowns.

Features include store ID, SKU ID, total price, base price, product display/feature status, and date (day, month, year).

Outputs real-time predicted units sold.

Dataset

The dataset from Kaggle contains three files:

train.csv → Historical sales data used to train the model.

test.csv → Used for testing or validation (optional in app).

sample_submission.csv → Contains sample structure for predictions.

Note: The train.csv has already been preprocessed and split, making model training more straightforward.

Features

Data cleaning and outlier removal to improve prediction accuracy.

Feature engineering from the week column (day, month, year).

Model training using Random Forest Regressor.

Streamlit app for real-time predictions with interactive dropdowns
