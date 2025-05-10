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
    
    # Кодирование категориальных признаков
    categories = df.select_dtypes(include=('object')).columns
    for col in categories:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df

def train_and_log_model():
    """Обучение модели с логированием в MLflow"""
    # Загрузка данных
    df = load_and_preprocess_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Параметры модели
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42
    }
    
    # Начало трекинга в MLflow
    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_params(params)
        
        # Обучение модели
        model_titanic = RandomForestClassifier(**params)
        model_titanic.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model_titanic.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
        
        # Логирование метрик
        mlflow.log_metrics(metrics)
        
        # Логирование модели с сигнатурой (типы входных/выходных данных)
        signature = infer_signature(X_train, model_titanic.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model_titanic,
            artifact_path="titanic_model",
            signature=signature,
            input_example=X_train.iloc[:5]  # Пример входных данных
        )
        
        # Логирование тестовых данных как артефакт
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv("test_data_titanic.csv", index=False)
        mlflow.log_artifact("test_data_titanic.csv")
        
        print(f"Метрики модели: {metrics}")

if __name__ == "__main__":
    # Установка URI для MLflow (может быть локальным или удаленным сервером)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # По умолчанию для локального UI
    
    # Установка имени эксперимента
    mlflow.set_experiment("Titanic_Survival_Prediction")
    
    train_and_log_model()

# def evaluate_model():
#     """Оценка модели"""
#     # Загрузка модели и тестовых данных
#     with open('/tmp/model.pkl', 'rb') as f:
#         model = pickle.load(f)
    
#     test_data = pd.read_csv('/tmp/test_data.csv')
#     X_test = test_data.drop('Survived', axis=1)
#     y_test = test_data['Survived']
    
#     y_pred = model.predict(X_test)
    
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred),
#         'roc_auc': roc_auc_score(y_test, y_pred)
#     }
    
#     # Сохранение метрик в файл
#     with open('/tmp/metrics.txt', 'w') as f:
#         f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
#         f.write(f"F1 Score: {metrics['f1']:.4f}\n")
#         f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
    
#     return metrics