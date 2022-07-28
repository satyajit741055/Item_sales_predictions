
from sales.logger import logging
from sales.exceptions import SalesException
from sales.constant import * 

from sales.entity.config_entity import DataTransformationConfig
from sales.entity.artifcat_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from sales.util.util import read_yaml_file,save_numpy_array_data,save_object

import os,sys
import pandas as pd 
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                    data_ingestion_artifact :DataIngestionArtifact,
                    data_transformation_config :DataTransformationConfig ):
        try:
            logging.info(f"{'>>'*30} Data Transformation Started {'<<'*30}")
            self.data_validation_artifact = data_validation_artifact,
            self.data_validation_artifact = self.data_validation_artifact[0]
            self.data_ingestion_artifact = data_ingestion_artifact,
            self.data_ingestion_artifact = self.data_ingestion_artifact[0]
            self.data_transformation_config = data_transformation_config
                        
        except Exception as e:
            raise SalesException(e,sys) from e


    def get_dats_transformer_object(self):
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(schema_file_path)

            numerical_features = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_features = dataset_schema[CATEGORICAL_COLUMN_KEY]

            num_pipeline = Pipeline(steps=[
                                     ('imputer',KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)),
                                     ('scaler',StandardScaler())
                                    ])


            cat_pipeline = Pipeline(steps=[
                                    ('imputer',KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)),
                                    ('scaler',StandardScaler())
                                    ])

            preprocessing = ColumnTransformer([
                                    ('num_pipeline', num_pipeline, numerical_features),
                                    ('cat_pipeline', cat_pipeline, categorical_features),
            ])


            logging.info(f"Categorical columns : {categorical_features}")
            logging.info(f"Numerical columns : {numerical_features}")
            return preprocessing
        except Exception as e:
            raise SalesException(e,sys) from e
    

    # function to get numerical and categorical columns 
    def numerical_categorical_column(self,dataframe):
        '''
        This function returns the numerical and categorical column 
        return numerical_columns,categorical_columns
        '''
        numerical_columns = [i for i in dataframe.columns if  dataframe[i].dtype != 'O']
        categorical_columns = [i for i in dataframe.columns if  dataframe[i].dtype == 'O']
        return numerical_columns,categorical_columns

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


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            
            preprocessing_obj = self.get_dats_transformer_object()
            logging.info(f'Preprocessing object received')

            logging.info(f'obtaining Test and Train files and loading as pandas dataframe')

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            schema_file_path = self.data_validation_artifact.schema_file_path

            schema = read_yaml_file(schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]

            logging.info(f'Seperating input features and target feature from test and train')

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

            logging.info(f'Starting applying preprocessing object on both files ')

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            processod_train_df = pd.DataFrame(input_feature_train_array,columns=input_feature_train_df.columns)
            processod_test_df = pd.DataFrame(input_feature_test_array,columns=input_feature_test_df.columns)

            logging.info(f'To check operations performed correctly or not will save this dataframe')
            
            transformed_train_path = self.data_transformation_config.transformed_train_file
            transformed_test_path = self.data_transformation_config.transformed_test_file

            os.makedirs(transformed_train_path,exist_ok=True)

            os.makedirs(transformed_test_path,exist_ok=True)

            transformed_train_file_sorce = os.path.join(transformed_train_path,'train.csv')
            transformed_test_file_source = os.path.join(transformed_test_path,'test.csv')

            processod_train_df.to_csv(transformed_train_file_sorce,index=False)
            processod_test_df.to_csv(transformed_test_file_source,index=False)

            train_arr = np.c_[ input_feature_train_array, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            ## Checking train_arr and saving 

            processod_train_array_df = pd.DataFrame(train_arr,columns=['Item_Weight','Item_Visibility','Item_MRP','Outlet_age','Item_Identifier','Item_Fat_Content',
                                                                        'Outlet_Size',
                                                                        'Outlet_Location_Type',
                                                                        'Outlet_Type','Item_Outlet_Sales'])
            processod_test_array_df = pd.DataFrame(test_arr,columns=['Item_Weight','Item_Visibility','Item_MRP','Outlet_age','Item_Identifier','Item_Fat_Content',
                                                                        'Outlet_Size',
                                                                        'Outlet_Location_Type',
                                                                        'Outlet_Type','Item_Outlet_Sales'])


            transformed_train_arr_df_sorce = os.path.join(transformed_train_path,'train_array_df.csv')
            transformed_test_arr_df_source = os.path.join(transformed_test_path,'test_arry_df.csv')

            processod_train_array_df.to_csv(transformed_train_arr_df_sorce,index=False)
            processod_test_array_df.to_csv(transformed_test_arr_df_source,index=False)


            ## Cheking Complet and saved 
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)


            data_transformation_artificat = DataTransformationArtifact(
                                         is_transformed=True,
                                         transformed_test_file_path=transformed_test_file_path,
                                         transformed_train_file_path=transformed_train_file_path,
                                         preprocessed_object_file_path=preprocessing_obj_file_path,
                                         transformed_train_arr_df_sorce = transformed_train_arr_df_sorce,
                                         transformed_test_arr_df_source = transformed_test_arr_df_source,

                                         message="Data Transformation Successful"
            )


            logging.info(f'Data Transformation artifact : {data_transformation_artificat}')

            return data_transformation_artificat
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        
        logging.info(f'{">>"*30} Data Transformation completed {"<<"*30}')
