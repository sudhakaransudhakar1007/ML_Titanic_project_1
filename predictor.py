import joblib
import pandas as pd

pipeline = joblib.load("Model/pipeline.pkl")

def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = pipeline.predict(df)[0]
    probability =pipeline.predict_proba(df)[0].tolist()
    return {
        "predction": int(prediction),
        "probablity":{
        "No survival" : probability[0],
        "Survival": probability[1]
         }
    }
