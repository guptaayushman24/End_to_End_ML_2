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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Here I took only the top 5 columns which are highly correlated with the target column and I apply the feature selection technique in the Model_Training.ipynb file
            numerical_columns = ['curbweight', 'enginesize', 'horsepower', 'citympg', 'highwaympg']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preproccesing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(['price'], axis=1)
            target_feature_train_df = train_df['price']
            input_feature_test_df = test_df.drop(['price'], axis=1)
            target_feature_test_df = test_df['price']

            input_feature_train_arr = preproccesing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preproccesing_obj.transform(input_feature_test_df)



            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preproccesing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
