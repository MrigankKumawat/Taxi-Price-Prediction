🚕 Taxi Trip Price Prediction
📖 Overview

This project predicts taxi trip fares using machine learning models based on features such as trip distance, duration, passenger count, time of day, day of week, traffic conditions, and weather.
The project also includes a Flask API for real-time predictions.

📊 Dataset

Dataset used: taxi_trip_pricing.csv (~4k–5k rows).

Features include:

Trip_Distance_km

Time_of_Day

Day_of_Week

Passenger_Count

Traffic_Conditions

Weather

Trip_Duration_Minutes

Target: Trip_Price

⚙️ Workflow

Data Cleaning & Preprocessing (handled missing values, duplicates).

Exploratory Data Analysis (EDA) using Matplotlib & Seaborn.

Feature Engineering (distance category).

Model Training:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor (✅ final choice)

Model Evaluation using R², MAE, RMSE, Cross-Validation.

Deployment using Flask API (tested with Postman).

🖥️ Tech Stack

Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn

Flask

Postman
