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

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'proprocessor.pkl')
class CustomEncoder(BaseEstimator,TransformerMixin) :
    def __init__(self,columns) :
        self.columns = columns

    def fit(self,X,y=None) :
        return self

    def transform(self,X) :
        X_encoded = X.copy()
        X_encoded = X.copy()
        for col in self.columns:
            if (col == 'fueltype'):
                X_encoded[col] = X_encoded[col].map({'gas': 1, 'disel': 0})
            elif (col == 'aspiration'):
                X_encoded[col] = X_encoded[col].map({'turbo': 1, 'std': 0})
            elif (col == 'drivewheel'):
                X_encoded[col] = X_encoded[col].map({'fwd': 1, 'rwd': 2, '4wd': 3})
            elif (col == 'enginelocation'):
                X_encoded[col] = X_encoded[col].map({'front': 1, 'rear': 0})
            elif (col=='carbody') :
                ordinal_label_carbody = {k:i for i,k in enumerate(X_encoded[col].unique(),0)}
                X_encoded[col] = X_encoded[col].map(ordinal_label_carbody)

            elif (col=='enginetype') :
                ordinal_label_enginetype = {k: i for i, k in enumerate(X_encoded[col].unique(), 0)}
                X_encoded[col] = X_encoded[col].map(ordinal_label_enginetype)
            elif (col=='fuelsystem') :
                    ordinal_label_fuelsystem = {k: i for i, k in enumerate(X_encoded[col].unique(), 0)}
                    X_encoded[col] = X_encoded[col].map(ordinal_label_fuelsystem)
            return X_encoded



        return X_encoded

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function will responsible for the data transformation
        '''
        try:
            numerical_columns = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carlength', 'carwidth', 'carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
                'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
            categorical_columns = ['CarName', 'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                ]
            )

            # Encoding of the categorical columns
            # lst_category_two_label = ['fueltype', 'aspiration', 'drivewheel', 'enginelocation']
            # def encoding(df, columns):
            #     for col in columns:
            #         if col == 'fueltype':
            #             df[col] = df[col].map({'gas': 1, 'disel': 0})
            #         elif col == 'aspiration':
            #             df[col] = df[col].map({'turbo': 1, 'std': 0})
            #         elif col == 'drivewheel':
            #             df[col] = df[col].map({'fwd': 1, 'rwd': 2, '4wd': 3})
            #         elif col == 'enginelocation':
            #             df[col] = df[col].map({'front': 1, 'rear': 0})
            #     return df
            logging.info('function 1  is completed')

            # lst_category_encoding_label = ['carbody','enginetype']
            # def encoding2(df, columns):
            #     for col in columns:
            #         if col == 'carbody':
            #             ordinal_label_carbody = {k: i for i, k in enumerate(df[col].unique(), 0)}
            #             df[col] = df[col].map(ordinal_label_carbody)
            #         elif col == 'enginetype':
            #             ordinal_label_enginetype = {k: i for i, k in enumerate(df[col].unique(), 0)}
            #             df[col] = df[col].map(ordinal_label_enginetype)
            #         elif col == 'fuelsystem':
            #             ordinal_label_fuelsystem = {k: i for i, k in enumerate(df[col].unique(), 0)}
            #             df[col] = df[col].map(ordinal_label_fuelsystem)
            #     return df
            # logging.info('function 2 is completed')

            # Creating the dictionary of the pipelines
            preprocessor = ColumnTransformer(
        transformers=[
            ('num_pipeline', num_pipeline, numerical_columns),
            ('cat_pipeline', cat_pipeline, categorical_columns),
            ('custom_encoder',CustomEncoder,categorical_columns)
        ]
    )
            logging.info('Preprocessor completed')

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path,test_path) :
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preproccesing_obj = self.get_data_transformer_object()

            logging.info('Reading of training and testing of the dataset is completed')
            logging.info('Obtaining the preprocessing object')

            target_column_name = "price"
            # drop_column_name = "CarName"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            logging.info(target_feature_test_df.head(5))

            input_feature_train_arr = preproccesing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preproccesing_obj.transform(input_feature_test_df)

            logging.info('Transformation Done')

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preproccesing_obj

            )
            logging.info('Data initiation is started')
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e :
            raise CustomException(e,sys)