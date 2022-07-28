from sales.logger import logging
from sales.exceptions import SalesException
from sales.entity.config_entity import ModelEvaluationConfig
from sales.entity.artifcat_entity import DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,DataTransformationArtifact
from sales.constant import *
import numpy as np
import os
import sys
from sales.util.util import write_yaml_file, read_yaml_file, load_object,load_data
from sales.entity.model_factory import evaluate_regression_model
import pandas as pd 



class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SalesException(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise SalesException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise SalesException(e, sys) from e

    def encoding(self,dataframe):
        

        # Changing Values of item Identifier because sales of item identifier are more so giving high value
        Item_Identifier = {'DR': 1, 'FD': 3, 'NC': 2}
        # Changing Values of Item_Fat_Content because sales of Regular are more so giving high value
        Item_Fat_Content = {'Low Fat': 1, 'Non Edible': 0, 'Regular': 2}
        # Changing Values of outlet size because sales in medium size outlets are more so giving high value
        Outlet_Size = {'High': 2, 'Medium': 3, 'Small': 1, np.nan: 0} 
        Outlet_Location_Type = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
        Outlet_Type = {'Grocery Store': 0,
        'Supermarket Type1': 2,
        'Supermarket Type2': 1,
        'Supermarket Type3': 3}
        for i in dataframe.columns:
            if i == "Item_Identifier":
                dataframe[i] = dataframe[i].map(Item_Identifier)
            elif i =="Item_Fat_Content":
                dataframe[i] = dataframe[i].map(Item_Fat_Content)
            elif i =="Outlet_Size":
                dataframe[i] = dataframe[i].map(Outlet_Size)
            elif i =="Outlet_Location_Type":
                dataframe[i] = dataframe[i].map(Outlet_Location_Type)
            elif i =="Outlet_Type":
                dataframe[i] = dataframe[i].map(Outlet_Type)
            else: 
                print("Not Able to Encode")
        
        return dataframe

    # function to get numerical and categorical columns 
    def numerical_categorical_column(self,dataframe):
        '''
        This function returns the numerical and categorical column 
        return numerical_columns,categorical_columns
        '''
        numerical_columns = [i for i in dataframe.columns if  dataframe[i].dtype != 'O']
        categorical_columns = [i for i in dataframe.columns if  dataframe[i].dtype == 'O']
        return numerical_columns,categorical_columns

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path

            train_df = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path,
                                                           )
            test_df = load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path,
                                                          )
            schema_content = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema_content[TARGET_COLUMN_KEY]

            input_feature_train_df = train_df.drop([target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop([target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            #Replacing same values with different names 
            input_feature_train_df['Item_Fat_Content'] = input_feature_train_df['Item_Fat_Content'].replace(['LF','low fat','reg'],['Low Fat','Low Fat','Regular'])
            input_feature_test_df['Item_Fat_Content'] = input_feature_test_df['Item_Fat_Content'].replace(['LF','low fat','reg'],['Low Fat','Low Fat','Regular'])

            #Removing unwanted data from name item_identifier
            input_feature_train_df['Item_Identifier'] = input_feature_train_df['Item_Identifier'].apply(lambda x:x[:2])
            input_feature_test_df['Item_Identifier'] = input_feature_test_df['Item_Identifier'].apply(lambda x:x[:2])

            # Feature Engineering of Outlet Establishment year
            input_feature_train_df['Outlet_age'] = 2013 - input_feature_train_df['Outlet_Establishment_Year']
            input_feature_train_df.drop(columns=['Outlet_Establishment_Year'],inplace=True)

            input_feature_test_df['Outlet_age'] = 2013 - test_df['Outlet_Establishment_Year']
            input_feature_test_df.drop(columns=['Outlet_Establishment_Year'],inplace=True)

            #Those product who are non-consumbale but have fat content will replace them with non-ediable fat content

            input_feature_train_df.loc[input_feature_train_df['Item_Identifier']=='NC','Item_Fat_Content'] = 'Non Edible'
            input_feature_test_df.loc[input_feature_test_df['Item_Identifier']=='NC','Item_Fat_Content'] = 'Non Edible'

            
            # dropping unwanted columns will work on it after some time 
            input_feature_train_df.drop(columns=['Item_Type','Outlet_Identifier'],inplace = True)
            input_feature_test_df.drop(columns=['Item_Type','Outlet_Identifier'],inplace = True)

            numerical_features_train,categorical_features_train = self.numerical_categorical_column(input_feature_train_df)
            numerical_features_test,categorical_features_test = self.numerical_categorical_column(input_feature_train_df)

            train_num_df = input_feature_train_df[numerical_features_train]
            train_cat_df = input_feature_train_df[categorical_features_train]
            train_cat_df = self.encoding(train_cat_df)
            train_cat_df['Outlet_Size'].replace(3,np.nan,inplace=True)

            test_num_df = input_feature_test_df[numerical_features_test]
            test_cat_df = input_feature_test_df[categorical_features_test]
            test_cat_df = self.encoding(test_cat_df)
            test_cat_df['Outlet_Size'].replace(3,np.nan,inplace=True)

            input_feature_train_df = pd.concat([train_num_df,train_cat_df],axis=1)
            input_feature_test_df = pd.concat([test_num_df,test_cat_df],axis=1)

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_df[target_column_name])
            test_target_arr = np.array(test_df[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")
            
            
            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]

            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                               X_train=input_feature_train_df,
                                                               y_train=train_target_arr,
                                                               X_test=input_feature_test_df,
                                                               y_test=test_target_arr,
                                                               base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                               )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise SalesException(e, sys) from e
            

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")