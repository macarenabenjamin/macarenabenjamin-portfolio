from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from textwrap import dedent
import pendulum
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score, ConfusionMatrixDisplay, precision_score, RocCurveDisplay)
import  joblib
import matplotlib.pyplot as plt
import os

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import datetime

from sklearn import datasets

from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.metrics import DatasetMissingValuesMetric
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase
#from evidently.ui.workspace.cloud import CloudWorkspace


default_args={
    "owner": "Macarena Benjamin",
    #"start_date": airflow.utils.dates.days_ago(2),
    #"end_date": datetime(),
    #"depends_on_past": False,
    #"email": ["airflow@example.com"],
    #"email_on_failure": False,
    #"email_on_retry": False,
    # If a task fails, retry it once after waiting at least 5 minutes
    #"retries": 1,
    #"retry_delay": timedelta(minutes=5),
    #"queue": "bash_queue",
    #"pool": "backfill",
    #"priority_weight": 10,
    #"end_date": datetime(2016, 1, 1),
    #"wait_for_downstream": False,
    #"sla": timedelta(hours=2),
    #"execution_timeout": timedelta(seconds=300),
    #"on_failure_callback": some_function,
    #"on_success_callback": some_other_function,
    #"on_retry_callback": another_function,
    #"sla_miss_callback": yet_another_function,
    #"trigger_rule": "all_success"
    }
    
# [END default_args]

# [START instantiate_dag]

with DAG(
    "macarenabenjamin_credit_risk",
    default_args=default_args,
    description="macarenabenjamin_credit_risk",
    schedule=None,
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
) as dag:
    
    # [END instantiate_dag]

    # [START documentation]
    
    dag.doc_md = __doc__
    
    # [END documentation]

    # [START extract_function]
    
    def extract(**kwargs):
        ti = kwargs["ti"]
        df = pd.read_csv("/root/airflow/dags/macarenabenjamin/df_min.csv",sep=",")
        ti.xcom_push(key="df", value=df.to_json())
      
    # [END extract_function]

    # [START transform_function]
    
    def transform(**kwargs):
        ti = kwargs["ti"]
        df = pd.read_json(ti.xcom_pull(key="df"))
        df = df.sort_values(by=["id", "loan_date"])
        df = df.reset_index(drop=True)
        df["loan_date"] = pd.to_datetime(df.loan_date)

        # Number of loans granted to a given user, before the current loan.
        df_grouped = df.groupby("id")
        df["nb_previous_loans"] = df_grouped["loan_date"].rank(method="first") - 1

        # Average amount of loans granted to a user, before the current loan.
        avg_amount_loans_previous = pd.Series()
        for user in df.id.unique():
            df_user = df.loc[df.id == user, :]
            avg_amount_loans_previous = avg_amount_loans_previous._append(df_user["loan_amount"].astype(float).rolling(df_user.shape[0], min_periods=1).mean().shift(periods=1))
        df["avg_amount_loans_previous"] = avg_amount_loans_previous

        # User age in years.
        df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
        df["age"] = (pd.to_datetime("today").normalize() - df["birthday"]).dt.days // 365

        # Years the user has been in employment.
        df["job_start_date"] = pd.to_datetime(df["job_start_date"], errors="coerce")
        df["years_on_the_job"] = (pd.to_datetime("today").normalize() - df["job_start_date"]).dt.days // 365

        # Flag that indicates if the user has his own car.
        df["flag_own_car"] = df.flag_own_car.apply(lambda x : 0 if x == "N" else 1)
        ti.xcom_push(key="transform_df", value=df.to_json())
    
    # [END transform_function]

    # [START load_function]
    
    def load(**kwargs):
        ti = kwargs["ti"]
        df = pd.read_json(ti.xcom_pull(key="transform_df"))
        df = df[['id', 'age', 'years_on_the_job', 'nb_previous_loans', 'avg_amount_loans_previous', 'flag_own_car', 'status']]
        df.to_csv("/root/airflow/dags/macarenabenjamin/transform_dataset_credit_risk_min.csv", index=False)
    
    # [END load_function]

    # [START train_register_function]
    
    def train_register(**kwargs):
        ti = kwargs["ti"]
        df = pd.read_csv("/root/airflow/dags/macarenabenjamin/transform_dataset_credit_risk_min.csv")
        cust_df = df.copy()
        cust_df.fillna(0, inplace=True)
        Y = cust_df['status']
        cust_df.drop(['id','status'], axis=1, inplace=True)
        X = cust_df
        Y = Y.astype("int")
        X_balance, Y_balance = SMOTE().fit_resample(X, Y)
        X_balance = pd.DataFrame(X_balance, columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_balance, Y_balance, stratify=Y_balance, test_size=0.3, random_state = 123)
        model = RandomForestClassifier(n_estimators=5)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print("Accuracy Score is {:.5}".format(accuracy_score(y_test, y_predict)))
        print("Precision Score is {:.5}".format(precision_score(y_test, y_predict)))
        print("Recall Score is {:.5}\n".format(recall_score(y_test, y_predict)))
        cm = confusion_matrix(y_test, y_predict, labels=model.classes_)
        print(pd.DataFrame(cm))
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        cmd.plot()
        rcd = RocCurveDisplay.from_estimator(model, X_test, y_test)
        rcd.plot()
        joblib.dump(model, "/root/airflow/dags/macarenabenjamin/model_risk_min.pkl")
        ti.xcom_push(key="X", value=X.to_json())
    
    # [END train_register_function]

    # [START load_predict_function]
    
    def load_predict(**kwargs):
        model = joblib.load("/root/airflow/dags/macarenabenjamin/model_risk_min.pkl")
        d = {
            "0": [32, 12, 2, 120, 1],
            "1": [29, 2, 1, 100, 0]
        }
        
        # Prediction
        print("Start Prediction ***************************************************************")
        print("Prediction 1: " + str(model.predict([d["0"]])[0]))
        print("Prediction 2: " + str(model.predict([d["1"]])[0]))
        print("End Prediction  ****************************************************************")

    # [END load_predict_function]

    # [START deploy_function]
    
    def deploy_model(**kwargs):
        print("\nStarting Deploy...\n")
        os.system('python3 /root/airflow/dags/macarenabenjamin/mbenjamin_deploy.py')

    # [END deploy_function]

    # [START monitor_function]
    
    def monitor_model(**kwargs):
        ti = kwargs["ti"]
        X = pd.read_json(ti.xcom_pull(key="X"))
        print("Starting Monitor...\n")
        ws = Workspace.create("mbenjamin-Workspace")
        project = create_project(ws)
        #ws = CloudWorkspace(token="dG9rbgGLZJ0iWaZMpbeJQ04/Ob0sA7V6NgDo/8NGfa00CrzH8wBQJVlGs9SfIZ92plR1kDe9VCdO9XuHu6YtvwzDOcFaZW7aHTBeWYJY67rPDGcDgRCw1NTpk6Zg+ZV39ZoDN1tIvMIjKJHsO38tpU3IP8TAmo39rPc/",url="https://app.evidently.cloud")
        #project = ws.get_project("3659d8ad-25cf-4df1-aea3-7e9beb1da9d5")
        report = create_report(X, X)
        ws.add_report(project.id, report)
        os.system('evidently ui --workspace ./mbenjamin-Workspace/ --port 18080')
           
    def create_project(ws: Workspace):
        YOUR_PROJECT_NAME = "mbenjamin_Credit_Risk"
        YOUR_PROJECT_DESCRIPTION = "mbenjamin_Credit_Risk_Evaluation"
        project = ws.create_project(YOUR_PROJECT_NAME)
        project.description = YOUR_PROJECT_DESCRIPTION
        #project.dashboard.add_panel()
        project.save()
        return project
    
    def create_report(current,reference):
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
        ])
        data_drift_report.run(current_data=current, reference_data=reference, column_mapping=None)
        data_drift_report.show(mode='inline')
        return data_drift_report
        

    # [END monitor_function]

    # [START main_flow]
    
    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract,
    )
    extract_task.doc_md = dedent(
        """\
    #### Extract task
    
    """
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
    )
    transform_task.doc_md = dedent(
        """\
    #### Transform task
    
    """
    )

    load_task = PythonOperator(
        task_id="load",
        python_callable=load,
    )
    load_task.doc_md = dedent(
        """\
    #### Load task
    
    """
    )

    train_register_task = PythonOperator(
        task_id="train_register",
        python_callable=train_register,
    )
    train_register_task.doc_md = dedent(
        """\
    #### Train and register task
    
    """
    )

    load_predict_task = PythonOperator(
        task_id="load_predict",
        python_callable=load_predict,
    )
    load_predict.doc_md = dedent(
        """\
    #### Load and predict task
    
    """
    )

    deploy_model_task = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )
    deploy_model.doc_md = dedent(
        """\
    #### Deploy task
    
    """
    )

    monitor_model_task = PythonOperator(
        task_id="monitor_model",
        python_callable=monitor_model,
    )
    monitor_model.doc_md = dedent(
        """\
    #### Monitot task
    
    """
    )

    extract_task >> transform_task >> load_task >> train_register_task >> load_predict_task >> deploy_model_task
    load_predict_task >> monitor_model_task
    # [END main_flow]
