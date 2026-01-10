import os
import sys

import numpy as np 
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, roc_auc_score

from fraudsecurity.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train_res, y_train_res,X_test,y_test,models,param): 
    try:
        report = {}
        trained_models = {}
        for name, model in models.items():
    
            param_grid = param.get(name, {})

            if param_grid: 
               grid = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
               grid.fit(X_train_res, y_train_res)
               best_model = grid.best_estimator_
            else:  
               model.fit(X_train_res, y_train_res)
               best_model = model

        
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        
            recall = recall_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

            report[name] = {
            "Recall": recall,
            "ROC_AUC": roc
            }

            trained_models[name] = best_model

        return report,trained_models

    except Exception as e:
        raise CustomException(e, sys)  

def get_best_model(model_report: dict, roc_threshold: float, recall_threshold: float):
    try:
        best_model_name = None
        best_model_roc = 0
        best_model_recall = 0

        for name, scores in model_report.items():
            roc = scores.get("ROC_AUC", 0)
            recall = scores.get("Recall", 0)

            if roc >= roc_threshold and recall >= recall_threshold:
                if roc > best_model_roc:
                    best_model_name = name
                    best_model_roc = roc
                    best_model_recall = recall

        return best_model_name, best_model_roc, best_model_recall
    except Exception as e:
        raise CustomException(e, sys)
