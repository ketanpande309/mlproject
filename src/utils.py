# Here we will be implementing common functionalities required for our project
import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        dir_name = os.makedirs(dir_path,exist_ok=True)

        with open (file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(Xtrain,ytrain,Xtest,ytest,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]


            gv = GridSearchCV(model,para,cv=3)
            gv.fit(Xtrain,ytrain)

            model.set_params(**gv.best_params_)
            model.fit(Xtrain,ytrain)

            ytrain_pred = model.predict(Xtrain)
            ytest_pred = model.predict(Xtest)

            train_model_score = r2_score(ytrain,ytrain_pred)
            print('Train model score for ',model,'is: ',train_model_score)

            test_model_score = r2_score(ytest,ytest_pred)
            print('Test model score for ',model,'is: ',test_model_score)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return pickle.load(file_obj)
            
        except Exception as e:
            raise CustomException(e,sys)