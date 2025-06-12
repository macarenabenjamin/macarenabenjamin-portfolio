from fastapi import FastAPI, Query
from typing import List
import  joblib
import numpy as np
from pydantic import BaseModel
import uvicorn

app = FastAPI()

model = joblib.load("/root/airflow/dags/macarenabenjamin/model_risk_min.pkl")

class request_body(BaseModel):
    age : int
    years_on_the_job: float
    nb_previous_loans: float
    avg_amount_loans_previous: int
    flag_own_car: int

@app.post("/predict")
def predict(request_data: request_body):
   features = [[
       request_data.age,
       request_data.years_on_the_job,
       request_data.nb_previous_loans,
       request_data.avg_amount_loans_previous,
       request_data.flag_own_car
   ]]
   prediction = model.predict(features)[0]
   print(features)
   print("\n")
   print(prediction)
   return {
       'prediction': prediction.item()
   }

if __name__ == '__main__':
        uvicorn.run(app, host='92.113.32.96', port=23000)