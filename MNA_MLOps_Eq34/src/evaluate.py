import pandas as pd
import sys
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import os
import json
from dvclive import Live




def evaluate_model(model_path, X_test_path, y_test_path, output_path, model_type):
    #os.makedirs(os.path.dirname(f"plots/{model_type}_cm.json"), exist_ok=True)
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1, output_dict=True)
    cm = confusion_matrix(y_test, predictions)
   
    write_evaluation_report(output_path, report, cm, model_type)
    predictions_pd = pd.concat([y_test,pd.DataFrame(predictions)], names=['y_true', 'y_pred'], axis = 1)
   
    return predictions_pd

def write_evaluation_report(file_path, report, confusion_matrix, model_type):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(f"metrics/{model_type}_metrics.json"), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(str(report))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix))
    metrics = {
        f"Evaluate {model_type}" : {
            'accuracy' : report['accuracy'],
            'presicion' : report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score']
        }
    }
    metrics_file_nm = f"metrics/{model_type}_metrics.json"
    with open(metrics_file_nm, 'w') as metrics_file:
        metrics_file.write(json.dumps(metrics, indent=2) + '\n')
    
  
     
    
    

if __name__ == '__main__':
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]
    output_path = sys.argv[4]
    pred_path = sys.argv[5]
    model_type = sys.argv[6]
    pred = evaluate_model(model_path, X_test_path, y_test_path, output_path, model_type)
    pd.DataFrame(pred).to_csv(pred_path, index=False)