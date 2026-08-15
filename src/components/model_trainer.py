import os
import sys

from catboost import CatBoostRegressor
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.utils import evaluate_models, save_object
from sklearn.linear_model import LinearRegression

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Standardizing on Regressor models (assuming student performance prediction)
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting": CatBoostRegressor(verbose=False),
            }

            param_grids = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],  # Removed 'friedman_mse'
                    'splitter': ['best', 'random'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],  # Removed 'friedman_mse'
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Gradient Boosting": {
                    # Note: 'friedman_mse' IS still valid for GradientBoostingRegressor criterion, 
                    # but loss criteria use 'squared_error'
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['squared_error', 'friedman_mse']
                },
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "CatBoosting": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },  
          }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, parameters=param_grids
            )

            # Get best model score from evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score > 0.6")
            
            logging.info(f"Best found model on testing dataset: {best_model_name}")

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

       
          