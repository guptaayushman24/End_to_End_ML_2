import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from src.utils import save_object
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import evaluate_models
from sklearn.metrics import r2_score
@dataclass
class ModelTrainerConfig() :
    trained_model_file_path : str = os.path.join('artifacts','model.pkl')

class ModelTrainer() :
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array) :
        try :
            logging.info('Train and Test Split')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegressor': LinearRegression(),
                'Support Vector' : SVR(),
                'DecisionTree' : DecisionTreeRegressor(),
                'RandomForest' : RandomForestRegressor()
            }
            # Hyperparameter of RandomForest Regressor
            '''params={
               'DecisionTree' :{
                   'max_depth': [10, 20, 30, 40],  # Adjust the values as needed
                   'min_samples_split': [2, 5, 10],  # Adjust the values as needed
                   'min_samples_leaf': [1, 2, 4]
               },
               'RandomForest' :{
            'n_estimators': [100, 200,300],     # Number of trees in the forest
            'max_depth': [10,20,30,40],      # Maximum depth of each tree
            'min_samples_split': [2,5,10],    # Minimum samples required to split a node
            'min_samples_leaf': [1, 2, 4],      # Minimum samples required at each leaf node
            'max_features': [ 'sqrt', 'log2'],   # Number of features to consider for the best split
            'bootstrap': [True, False]          # Whether bootstrap samples are used
               },
             'Support Vector':{},
               'LinearRegressor':{}
           }'''




            model_report : dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if (best_model_score<0.6) :
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")
            # From the last log file we can see that Random Forest Regressor is the best model so we will apply hyperparameter tunning on Random Forest Regressor

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predictd = best_model.predict(X_test)
            r2_square_value = r2_score(y_test,predictd)
            logging.info('The name of the best model file is {}'.format(best_model))
            return r2_square_value

        except Exception as e :
            raise CustomException (e,sys)

    # Without hyperparameter tunning we are  getting the accuracy around 93% and finally we are chossing RandomForestRegressor() as the final model for our problem