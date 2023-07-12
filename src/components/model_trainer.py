# In this file, we will train our model using various algorithm and will check there performance using different different metrics
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
#from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting training and test data')
            Xtrain,ytrain,Xtest,ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Random Forest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),
                'CatBoosting Regressor':CatBoostRegressor(verbose=False),
                'Adaboost Regressor':AdaBoostRegressor(),
                'Knearest Regressor':KNeighborsRegressor()
            }

            model_report:dict = evaluate_models(Xtrain=Xtrain,ytrain=ytrain,Xtest=Xtest,ytest=ytest,models=models)

            #To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.7:
                raise CustomException('No best model found')
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(Xtest)
            r2_scores = r2_score(predicted,ytest)

            return r2_scores,best_model

        except Exception as e:
            raise CustomException(e,sys)

    