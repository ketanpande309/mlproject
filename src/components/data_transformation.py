#This file will contain data transformation, i.e cleaning the data and making it ready for model
import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Columns:{numerical_columns}")
            logging.info(f"Categorical Columns{categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('Fetching Train and Test Data')
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('Train and Test Data Obtained')

            logging.info('Object Preprocessing Object')
            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numerical_column = ['writing_score','reading_score']

            input_feature_train_data = train_data.drop(columns=[target_column],axis=1)
            target_train_data = train_data[target_column]

            input_feature_test_data = test_data.drop(columns=[target_column],axis=1)
            target_test_data = test_data[target_column]

            logging.info('Applying preprocessing on training and testing data')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[input_feature_train_arr,np.array(target_train_data)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_test_data)]

            logging.info('Saved Preprocessing Object')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
    
        except Exception as e:
            raise CustomException(e,sys)