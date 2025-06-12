from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import json

app = FastAPI()

from pydantic import BaseModel

class DataRisk(BaseModel):
  loan_id: object
  id: object
  code_gender: object
  flag_own_car: object
  flag_own_realty: object
  cnt_children: object
  amt_income_total: object
  name_income_type: object
  name_education_type: object
  name_family_status: object
  name_housing_type: object
  days_birth: object
  days_employed: object
  flag_mobil: object
  flag_work_phone: object
  flag_phone: object
  flag_email: object
  occupation_type: object
  cnt_fam_members: object
  birthday: object
  loan_date: object
  loan_amount: object
  age: object
  years_on_the_job: object
  avg_amount_loans_previous: object
  job_start_date: object

@app.get("/")
def hello():
    return {"API":"API is working fine"}

@app.post("/predict")
async def process_text(data: DataRisk):

    model = joblib.load("/root/python-projects/credit-risk/macarenabenjamin/pipeline")

    data = data.model_dump()

    columns = [
        "loan_id",
        "code_gender",
        "flag_own_car",
        "flag_own_realty",
        "cnt_children",
        "amt_income_total",
        "name_income_type",
        "name_education_type",
        "name_family_status",
        "name_housing_type",
        "days_birth",
        "days_employed",
        "flag_mobil",
        "flag_work_phone",
        "flag_phone",
        "flag_email",
        "occupation_type",
        "cnt_fam_members",
        "birthday",
        "loan_date",
        "loan_amount",
        "job_start_date",
        "age",
        "years_on_the_job",
        "avg_amount_loans_previous"
    ]

    datavalues = [
        data["loan_id"],
        data["code_gender"],
        data["flag_own_car"],
        data["flag_own_realty"],
        data["cnt_children"],
        data["amt_income_total"],
        data["name_income_type"],
        data["name_education_type"],
        data["name_family_status"],
        data["name_housing_type"],
        data["days_birth"],
        data["days_employed"],
        data["flag_mobil"],
        data["flag_work_phone"],
        data["flag_phone"],
        data["flag_email"],
        data["occupation_type"],
        data["cnt_fam_members"],
        data["birthday"],
        data["loan_date"],
        data["loan_amount"],
        data["age"],
        data["years_on_the_job"],
        data["avg_amount_loans_previous"],
        data["job_start_date"]
    ]

    df = pd.DataFrame([datavalues], columns=columns)

    return model.predict(df.iloc[0].to_frame().T)[0].item()

if __name__=="__main__":
    uvicorn.run(app, host="92.113.32.96", port=8000)