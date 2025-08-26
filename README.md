# 🚖 Taxi Trip Price Prediction

An end-to-end Machine Learning project to predict taxi fares based on trip details such as distance, time, traffic, weather, and passenger count.
It also includes a Flask API for real-time predictions.

## 📌 Features

✔️ Predicts taxi trip prices using Gradient Boosting Regressor

✔️ Data cleaning, preprocessing, and feature engineering

✔️ Interactive visualizations (box plots, violin plots, 
heatmaps, scatter plots)

✔️ Cross-validation for model reliability

✔️ Flask API for deployment and Postman testing

## 🛠️ Technologies Used

 - Python 🐍 – Core language
 - Pandas, NumPy – Data cleaning & analysis
 - Matplotlib, Seaborn – Visualizations
 - Scikit-learn – ML models & pipeline
 - Flask – API deployment
 - Postman – API testing

## 📊 Dataset

📁 File: taxi_trip_pricing.csv

🎯 Target Variable: Trip_Price

Features include:

- Trip Distance (km)
- Trip Duration (minutes)
- Passenger Count
- Time of Day
- Day of Week
- Traffic Conditions
- Weather

## 📁 Files Included

📓 Taxi-Price.ipynb – Jupyter Notebook (EDA + Model Training)

📄 taxi_trip_pricing.csv – Dataset

📂 vscode/ – Deployment & Model Export

     ├── interference.py – Model training & export script
     
     ├── server.py – Flask API for predictions
     
     ├── model.pkl – Trained ML model
     
     ├── pipeline.pkl – Preprocessing pipeline
     
📄 README.md – Project documentation
