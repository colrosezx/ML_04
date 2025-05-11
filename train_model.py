import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    df = pd.read_csv('Titanic.csv', delimiter=',')
    df = df.dropna()

    categories = df.select_dtypes(include=('object')).columns
    for col in categories:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df

def train_and_log_model():

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
        
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv("test_data_titanic.csv", index=False)
        mlflow.log_artifact("test_data_titanic.csv")
        
        print(f"Метрики модели: {metrics}")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    mlflow.set_experiment("Titanic_Survival_Prediction")
    
    train_and_log_model()
