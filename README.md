🚕 Taxi Trip Price Prediction
📖 Overview

This project predicts taxi trip fares using machine learning based on features like:

Trip distance

Trip duration

Passenger count

Time of day

Day of week

Traffic conditions

Weather

It also includes a Flask API for real-time predictions.

📊 Dataset

File: taxi_trip_pricing.csv
Target: Trip_Price
Features:

Trip_Distance_km

Trip_Duration_Minutes

Passenger_Count

Time_of_Day

Day_of_Week

Traffic_Conditions

Weather

⚙️ Workflow

Data Cleaning & Preprocessing (handled missing values, duplicates).

Exploratory Data Analysis (EDA) → Matplotlib & Seaborn.

Feature Engineering → created distance categories.

Model Training:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor (final choice ✅)

Model Evaluation → R², MAE, RMSE, Cross-Validation.

Deployment → Flask API, tested via Postman.

🖥️ Tech Stack

Python 🐍

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Flask

Postman
