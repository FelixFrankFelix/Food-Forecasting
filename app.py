import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Load data
fpn = pd.read_csv('raw_data.csv')  # Load your data here
#print(fpn.head())
# Function to create features
def create_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df.set_index('date',inplace =True)
    return df

# Function to train XGBoost model
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    reg = LinearRegression()
    #reg = xgb.XGBRegressor(booster='gbtree')
    reg.fit(X, y)#, verbose=100)
    return reg

# Define Streamlit app
st.title('Commodity Price Prediction App')

# Sidebar inputs
commodity_list = fpn['commodity'].unique().tolist()
selected_commodity = st.sidebar.selectbox('Select Commodity:', commodity_list)
predict_date = st.sidebar.date_input('Select Date for Prediction:', datetime.today())


# Filter data for selected commodity
data = fpn[fpn['commodity'] == selected_commodity][["date", "price"]]
data = data.groupby('date').price.mean().reset_index()


# Create features
data = create_features(data)

# Define features and target
FEATURES = ['quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
TARGET = 'price'

# Train XGBoost model
model = train_model(data)

# Generate predictions for the selected date
last_train_date = data.index.max()
print("last date",last_train_date)
# Convert last_train_date to datetime if it's not already
if not isinstance(last_train_date, pd.Timestamp):
    last_train_date = pd.to_datetime(last_train_date)
    print("last date trans",last_train_date)
# Generate predictions for the selected date
predict_dates = pd.date_range(start=last_train_date + timedelta(days=1), end=predict_date, freq='MS')
print("predict date:",predict_dates)
predictions = []

for date in predict_dates:
    features = create_features(pd.DataFrame({'date': [date]}))
    prediction = model.predict(features[FEATURES])[0]
    predictions.append((date, prediction))

# Display predictions
st.subheader('Predictions:')
predictions_df = pd.DataFrame(predictions, columns=['Date', 'Predicted Price'])
print(predictions_df.iloc[-1])
date_l = predictions_df.iloc[-1]["Date"]
pred_l = predictions_df.iloc[-1]["Predicted Price"]
rounded_pred_l = round(pred_l,2)
rounded_pred_l = str(rounded_pred_l)
st.success(f'The Prediction for {selected_commodity} in {date_l.day}/{date_l.month}/{date_l.year} is {rounded_pred_l} Naira per KG')

# Plot time series of prices with predictions
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['price'], color='blue', label='Historical Prices')
plt.plot(predictions_df['Date'], predictions_df['Predicted Price'], color='red', label='Predicted Prices')
plt.title('Commodity Price Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(plt)
