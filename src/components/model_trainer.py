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
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer() :
    def __init__(self) :
        self.model_trainer_config= ModelTrainerConfig()

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


            model_report : dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if (best_model_score<0.6) :
                raise CustomException("No best model foound")
            logging.info("Best found model on both training and testing dataset")

            predictd = best_model.predict(X_test)
            r2_square_value = r2_score(y_test,predictd)

            return r2_square_value

        except Exception as e :
            raise CustomException (e,sys)