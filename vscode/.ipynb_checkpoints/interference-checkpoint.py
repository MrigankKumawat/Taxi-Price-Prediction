import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attrbs, cat_attrbs):
    # For Numerical Columns
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown = "ignore")),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrbs),
        ("cat", cat_pipeline, cat_attrbs),
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("taxi_trip_pricing.csv")

    ## For Numeric Columns
    df['Trip_Distance_km'] = df['Trip_Distance_km'].fillna(df['Trip_Distance_km'].median())
    df['Trip_Price'] = df['Trip_Price'].fillna(df['Trip_Price'].median())
    df['Trip_Duration_Minutes'] = df['Trip_Duration_Minutes'].fillna(df['Trip_Duration_Minutes'].median())

    ## For Categorical Columns
    for col in ['Time_of_Day', 'Day_of_Week', 'Passenger_Count', 'Traffic_Conditions', 'Weather']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['distance_cat'] = pd.cut(df['Trip_Distance_km'], bins = [0, 25, 50, 75, 100, np.inf], labels = [1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_set, test_set in split.split(df, df['distance_cat']):
        strat_train_set = df.loc[train_set]
        strat_test_set = df.loc[test_set]

    taxi_train = strat_train_set.copy()
    taxi_features_train = taxi_train.drop(["Trip_Price", "distance_cat"], axis = 1)
    taxi_label_train = taxi_train['Trip_Price']
    num_attrbs_train = ['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes']
    cat_attrbs_train = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']

    pipeline = build_pipeline(num_attrbs_train, cat_attrbs_train)
    taxi_prepared_train = pipeline.fit_transform(taxi_features_train)
    model = GradientBoostingRegressor(random_state = 42)
    model.fit(taxi_prepared_train, taxi_label_train)

    joblib.dump(model, MODEL_FILE) 
    joblib.dump(pipeline, PIPELINE_FILE) 
    print("Model is Trained and Saved.")
