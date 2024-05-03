import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
fpn = pd.read_csv('raw_data.csv')


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to predict future prices
def predict_future_prices(df, end_month, end_year):
    # Create future date
    
    end_date = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    future_dates = pd.date_range(start=df.index.max(), end=end_date, freq='M')
    # Extend the index of df to include future date
    extended_index = df.index.union(future_dates)
    df_extended = df.reindex(extended_index)

    # Create a deterministic process
    dp = DeterministicProcess(
        index=df_extended.index,  # dates from the extended dataset
        constant=True,             # dummy feature for the bias (y_intercept)
        order=1,                   # the time dummy (trend)
        drop=True,                 # drop terms if necessary to avoid collinearity
    )

    # Create features for the extended dataset
    X_extended = dp.in_sample()

    # Target variable for the training dataset
    y = df["price"]

    # Initialize Linear Regression model
    model = LinearRegression(fit_intercept=False)
    
    # Fit the model on the training dataset
    model.fit(X_extended.loc[df.index], y)

    # Predict prices for the extended dataset (including future date)
    y_pred_extended = pd.Series(model.predict(X_extended), index=X_extended.index)

    # Extract the prediction for the future date
    future_prediction = y_pred_extended.loc[end_date]

    return y_pred_extended, future_prediction

# Function to plot actual and predicted prices
def plot_prices(actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='orange')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Main Streamlit app
def main():
    st.title('Food Price Prediction App')
    commodity_list = fpn['commodity'].unique().tolist()
    selected_commodity = st.selectbox('Select Commodity:', commodity_list)
    # Filter data for selected commodity
    data = fpn[fpn['commodity'] == selected_commodity][["date", "price"]]
    data = data.groupby('date').price.mean().reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    print(data)
    print(data.info())
    # User input for month and year
    # Dictionary to map month names to their corresponding numbers
    month_map = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
        }

    # User input for month and year
    end_month_name = st.selectbox('Select month', options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], index=11)
    end_month = month_map[end_month_name]  # Convert month name to number
    end_year = st.number_input('Enter year', value=int(data.index.year.max()))

    future_predictions, future_price = predict_future_prices(data, end_month, int(end_year))
    # Plot actual and predicted prices
    plot_prices(data['price'], future_predictions)

    # Print predicted price
    st.write(f'Predicted price for {end_month_name} {end_year}: {future_price:.2f} Naira Per KG')

if __name__ == '__main__':
    main()
