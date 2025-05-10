from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import os
from airflow.models import Variable

# Путь относительно расположения DAG-файла
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
csv_path = os.path.join(root_dir, 'Titanic.csv')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'titanic_ml_pipeline',
    default_args=default_args,
    description='ML Pipeline for Titanic survival prediction',
    schedule_interval='@weekly',
)

def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    df = pd.read_csv(csv_path, delimiter=',')
    df = df.dropna()
    
    categories = df.select_dtypes(include=('object')).columns
    for col in categories:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df

def train_and_log_model(**kwargs):
    """Обучение модели с логированием в MLflow"""

    df = load_and_preprocess_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42
    }
    
    with mlflow.start_run():

        mlflow.log_params(params)
        
        model_titanic = RandomForestClassifier(**params)
        model_titanic.fit(X_train, y_train)

        y_pred = model_titanic.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
        
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, model_titanic.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model_titanic,
            artifact_path="titanic_model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )


preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=load_and_preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_and_log_model',
    python_callable=train_and_log_model,
    dag=dag,
)

preprocess_data_task >> train_model_task