import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# Title
st.title("📈 Stock Price Prediction Using Machine Learning")
st.write("Apple (AAPL) Stock Prediction Dashboard using Linear Regression and LSTM")


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")
    return df


df = load_data()


# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date")


# Dataset Preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


st.write("Dataset Shape:", df.shape)
st.write("Columns:", list(df.columns))


# Stock history graph
st.subheader("📈 AAPL Closing Price History")


fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    df["Date"],
    df["Close"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Closing Price ($)")
ax.set_title("Apple Stock Closing Price")

plt.xticks(rotation=45)

st.pyplot(fig)



# ==============================
# Linear Regression Model
# ==============================

st.subheader("🤖 Linear Regression Model")


X = df[["Open","High","Low","Volume"]]

y = df["Close"]


split = int(len(df)*0.8)


X_train = X[:split]
X_test = X[split:]


y_train = y[:split]
y_test = y[split:]


lr_model = LinearRegression()


lr_model.fit(
    X_train,
    y_train
)


lr_prediction = lr_model.predict(X_test)



# Metrics

lr_mae = mean_absolute_error(
    y_test,
    lr_prediction
)


lr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lr_prediction
    )
)


lr_r2 = r2_score(
    y_test,
    lr_prediction
)



col1,col2,col3 = st.columns(3)


with col1:
    st.metric(
        "MAE",
        round(lr_mae,4)
    )


with col2:
    st.metric(
        "RMSE",
        round(lr_rmse,4)
    )


with col3:
    st.metric(
        "R² Score",
        round(lr_r2,4)
    )



# Actual vs predicted graph

st.write("### Actual vs Predicted Close Price")


fig2,ax2 = plt.subplots(figsize=(12,5))


ax2.plot(
    y_test.values[:100],
    label="Actual"
)


ax2.plot(
    lr_prediction[:100],
    label="Predicted"
)


ax2.legend()

st.pyplot(fig2)



# ==============================
# LSTM MODEL
# ==============================


st.subheader("🧠 LSTM Deep Learning Model")



# Load model

@st.cache_resource
def load_lstm_model():

    model = load_model(
        "AAPL_LSTM_Model.h5"
    )

    return model



lstm_model = load_lstm_model()



# Load scaler

with open("scaler.pkl","rb") as file:

    scaler = pickle.load(file)



# User Input Section

st.subheader("Enter Latest Closing Price")


user_price = st.number_input(
    "Enter today's closing price ($)",
    value=150.0
)



if st.button("Predict Using LSTM"):


    input_data = np.array(
        [[user_price]]
    )


    scaled_input = scaler.transform(
        input_data
    )


    X_input = np.repeat(
        scaled_input,
        60
    )


    X_input = X_input.reshape(
        1,
        60,
        1
    )


    prediction = lstm_model.predict(
        X_input
    )


    final_prediction = scaler.inverse_transform(
        prediction
    )


    st.success(
        f"Predicted Next Closing Price: ${final_prediction[0][0]:.2f}"
    )



# Automatic prediction using last 60 days

close_prices = df[["Close"]].values


scaled_close = scaler.transform(
    close_prices
)


last_60_days = scaled_close[-60:]


X_lstm = last_60_days.reshape(
    1,
    60,
    1
)


lstm_prediction = lstm_model.predict(
    X_lstm
)


predicted_price = scaler.inverse_transform(
    lstm_prediction
)



st.write(
    "Automatic prediction from latest data:"
)


st.success(
    f"${predicted_price[0][0]:.2f}"
)




# Recent prices graph

st.subheader("Last 60 Days Closing Price")


fig3,ax3 = plt.subplots(figsize=(12,5))


ax3.plot(
    df["Close"].tail(60)
)


ax3.set_xlabel("Days")
ax3.set_ylabel("Closing Price ($)")


st.pyplot(fig3)



# ==============================
# Model Comparison
# ==============================


st.subheader("📊 Model Comparison")


comparison = pd.DataFrame({

    "Model":[
        "Linear Regression",
        "LSTM"
    ],


    "Prediction":[
        round(lr_prediction[-1],2),
        round(predicted_price[0][0],2)
    ],


    "R2 Score":[
        round(lr_r2,4),
        "Deep Learning"
    ]

})


st.dataframe(comparison)



# Project Summary

st.subheader("📌 Project Summary")


st.write(
"""
This application predicts Apple (AAPL) stock closing prices.

Models used:

1. Linear Regression
- Uses Open, High, Low and Volume.
- Provides statistical prediction.

2. LSTM Deep Learning
- Uses previous 60 days stock prices.
- Learns time-series patterns.

The models are compared based on performance.
"""
)
