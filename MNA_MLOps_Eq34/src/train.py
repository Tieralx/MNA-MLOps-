import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import yaml
import sys
import joblib



def train_model(X_train_path, y_train_path, model_type):
    
    if model_type == 'lr': 
        model_params = config_params["grid_search"]["lr_params"]
        model_train = LogisticRegression(**model_params)
    elif model_type == 'clf':
        model_params = config_params["grid_search"]["clf_params"]
        model_train = DecisionTreeClassifier(**model_params)
    elif model_type == 'xgb': 
        model_params = config_params["grid_search"]["xgb_params"]
        model_train= xgb.XGBClassifier(**model_params) 
    
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    mlflow.set_tracking_uri(uri=config_params["mlflow_config"]["uri"])
    mlflow.set_experiment(config_params["mlflow_config"]["experiment_name"])
    
    with mlflow.start_run():
        model_train.fit(X_train, y_train)
        mlflow.sklearn.log_model(model_train, f"{model_type}_model")

    return model_train

if __name__ == "__main__":
    yaml_file = sys.argv[1]
    X_train_path = sys.argv[2]
    y_train_path = sys.argv[3]
    model_type = sys.argv[4]
    with open(yaml_file, 'r') as yaml_file:
        config_params = yaml.safe_load(yaml_file)
    #model_dir = config_params['data']['models']
    model_path = f"models/{model_type}_model.pkl"

    model = train_model(X_train_path, y_train_path, model_type)
    
    joblib.dump(model, model_path)
    
    
    
    
    
    
    
        
    