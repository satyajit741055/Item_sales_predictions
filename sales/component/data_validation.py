from sales.logger import logging
from sales.exceptions import SalesException
from sales.constant import * 
from sales.entity.config_entity import DataValidationConfig
from sales.entity.artifcat_entity import DataIngestionArtifact, DataValidationArtifact
from sales.util.util import read_yaml_file

import os,sys

# Eveidently is used for Data Drigitign etc 
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

import pandas as pd 
import json


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            logging.info(f"{'>>'*30} DataValidation started {'<<'*30} \n\n")
            self.data_ingestion_artifact = data_ingestion_artifact,
            self.data_ingestion_artifact = self.data_ingestion_artifact[0]
            
            self.data_validation_config = data_validation_config,
            self.data_validation_config = self.data_validation_config[0]
            
            self.current_time_stamp = CURRENT_TIME_STAMP

        except Exception as e:
            raise SalesException(e,sys) from e

    def get_train_test_dataset(self):
        try:

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            return train_df,test_df 
        except Exception as e:
            raise SalesException(e,sys) from e

    def is_train_test_exist(self):
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

        
            is_file_exist = os.path.exists(train_file_path) and os.path.exists(test_file_path)

            logging.info(f"IS train and test File Exists? -> {is_file_exist}")


            return is_file_exist
        except Exception as e:
            raise SalesException(e,sys) from e
        
    def validation_dataset_schema(self) ->bool:
        try:
            validations_status = False

            schema_file = read_yaml_file(self.data_validation_config.schema_file_path)
            schema_dict = schema_file[DATA_VALIDATION_SCHEMA_KEY]
            train_df , test_df = self.get_train_test_dataset()
            

            for column,data_type in schema_dict.items():
                train_df[column].astype(data_type)
                test_df[column].astype(data_type)


            validations_status = True

            logging.info("Data Validation done and statys : {Validations_status}")
            return validations_status
        except Exception as e:
            raise SalesException(e,sys) from e


    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])

            train_df ,test_df = self.get_train_test_dataset()
            test_df['Item_Outlet_Sales'] = 0 # 'Item_Outlet_Sales' not present in test so profile.calculate giving error 
            profile.calculate(train_df,test_df)

            report = json.loads(profile.json())

            report_file_path = self.data_validation_config.report_file_path

            report_dir = os.path.dirname(report_file_path)
            
            os.makedirs(report_dir,exist_ok=True)

            with open(report_file_path,'w') as report_file:
                json.dump(report,report_file,indent=6)

            
            return report

        except Exception as e:
            raise SalesException(e,sys) from e


    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df ,test_df = self.get_train_test_dataset()
            test_df['Item_Outlet_Sales'] = 0 # 'Item_Outlet_Sales' not present in test so profile.calculate giving error 
            dashboard.calculate(train_df ,test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_paage_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_paage_dir,exist_ok=True)

            dashboard.save(report_page_file_path)
        except Exception as e:
            raise SalesException(e,sys) from e

    def is_data_drif_found(self) -> bool:
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.is_train_test_exist()
            is_validated = self.validation_dataset_schema()
            self.is_data_drif_found()

            data_validation_artificat = DataValidationArtifact(
                                         schema_file_path=self.data_validation_config.schema_file_path,
                                         report_file_path=self.data_validation_config.report_file_path,
                                         report_page_file_path=self.data_validation_config.report_page_file_path,
                                         is_validated= is_validated,
                                         message=" data Validation Performed Successfully"
            )



            logging.info(f'Data validation artifact : {data_validation_artificat}')

            return data_validation_artificat
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info(f'{">>"*30} Data Validation completed {"<<"*30} \n\n')