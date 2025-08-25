from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np

from flask import Flask
app = Flask(__name__)

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/predict", methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        processed_data = pipeline.transform(df)
        prediction = model.predict(processed_data)

        return jsonify({"Predicted Price:": prediction[0]})
    
    except Exception as e:
        return jsonify({"Error": str(e)})

app.run(debug = True)