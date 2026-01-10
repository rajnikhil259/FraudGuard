import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from fraudsecurity.exception import CustomException
from fraudsecurity.logger import logging
from fraudsecurity.utils import save_object, evaluate_models, get_best_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Applying SMOTE on training data")

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            logging.info("Training data balanced using SMOTE")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "K-Neighbors": KNeighborsClassifier(),
               "XGBoost": XGBClassifier(
                          objective="binary:logistic",
                          eval_metric="auc",
                          use_label_encoder=False,
                          random_state=42
                        ),
                "CatBoost": CatBoostClassifier(verbose=False, random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "Naive Bayes": GaussianNB()
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5]
                },
                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.8, 1]
                },
                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [200, 500]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                },
                "SVM": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },
                "Naive Bayes": {
                    "var_smoothing": [1e-09, 1e-08, 1e-07]
                }
            }

            logging.info("Evaluating models with hyperparameter tuning")

            model_report, trained_models = evaluate_models(
                X_train_res=X_train_res,
                y_train_res=y_train_res,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            roc_threshold = 0.75
            recall_threshold = 0.40

            best_model_name, best_model_roc, best_model_recall = get_best_model(
                model_report,
                roc_threshold=roc_threshold,
                recall_threshold=recall_threshold
            )

            if best_model_name is None:
                logging.info(
                    "No model satisfied the ROC-AUC and Recall thresholds. "
                    "Consider further tuning."
                )
                return None, None, None

            best_model = trained_models[best_model_name]

            logging.info(
                f"Best Model Selected: {best_model_name} | "
                f"ROC_AUC: {best_model_roc:.4f} | "
                f"Recall: {best_model_recall:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return best_model_name, best_model_roc, best_model_recall

        except Exception as e:
            raise CustomException(e, sys)
