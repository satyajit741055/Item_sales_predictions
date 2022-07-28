'''from sales.config.configuration import Configuration

from sales.constant import *


from sales.entity.config_entity import DataIngestionConfig,DataValidationConfig
from sales.entity.artifcat_entity import DataIngestionArtifact,DataValidationArtifact

from sales.component.data_validation import DataValidation
from sales.component.data_ingestion import DataIngestion
import pandas as pd

a = Configuration(CONFIG_FILE_PATH,CURRENT_TIME_STAMP)

# print(a.get_data_ingestion_config())

b = DataIngestion(a.get_data_ingestion_config())

data_ingestion_artifact = b.initiate_data_ingestion()

print(f'Dataingestion {data_ingestion_artifact}')

a = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=DataValidationConfig)
train_df , test_df = a.get_train_test_dataset()

print(train_df.shape)

from pyparsing import col
from xgboost import train
from sales.entity.model_selector import ModelSelctor
import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(r"D:\Projects_new\Stores_Sales_Prediction\sales\artifact\data_transformation\2022-07-25-15-23-29\preprocessed_files\train_transformed\train_array_df.csv")

x = train_df.drop(columns=['Item_Outlet_Sales'])
y = train_df['Item_Outlet_Sales']
x_train, y_train, x_test, y_test = train_test_split(x,y,test_size=0.20)
base_accuracy = 0.5 


model = ModelSelctor(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,base_accuracy=0.4)

model , model_name = model.get_best_model()

print(f'model { model}')
print(f'model_name {model_name}')

'''
from flask import Flask, request
import sys

import pip
from sales.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from sales.logger import logging
from sales.exceptions import SalesException
import os, sys
import json
from sales.config.configuration import Configuration
from sales.constant import CONFIG_DIR, get_current_time_stamp
from sales.pipeline.pipeline import Pipeline
from sales.entity.sales_predictor import salesPredictor, SalesData
from flask import send_file, abort, render_template
ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "sales"
SAVED_MODELS_DIR_NAME = "saved_models"
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

'''
sales_data = SalesData(Item_Identifier="FDI28",
                                   Item_Weight=14.3,
                                   Item_Fat_Content="Low Fat",
                                   Item_Visibility=0.263,
                                   Item_Type="Frozen Foods",
                                   Item_MRP=79.4302,
                                   Outlet_Establishment_Year=1987,
                                   Outlet_Size="High",
                                   Outlet_Location_Type="Tier 3",
                                   Outlet_Type = "Supermarket Type1",
                                   Item_Outlet_Sales = "None", 
                                   Outlet_Identifier = "OUT013"
                                   )

sales_predictor = salesPredictor(model_dir=MODEL_DIR)
sales_df = sales_data.get_sales_input_data_frame()
Item_Outlet_Sales = sales_predictor.predict(X=sales_df)
print(Item_Outlet_Sales)'''


from flask import Flask, request
from sales.logger import logging,get_log_dataframe
from sales.config.configuration import Configuration
from sales.constant import CONFIG_DIR, get_current_time_stamp
from sales.pipeline.pipeline import Pipeline
from sales.entity.sales_predictor import salesPredictor, SalesData
from flask import send_file, abort, render_template
import os





ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "sales"
SAVED_MODELS_DIR_NAME = "saved_models"
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)




SALES_DATA_KEY = "Sales_data"
ITEM_OUTLET_SALES_KEY = "Item_Outlet_Sales"




def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.run_pipeline()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


train()