
import os
import sys

from sales.exceptions import SalesException
from sales.util.util import load_object
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class SalesData:

    def __init__(self,
                    Item_Identifier: str,
                    Item_Weight: float,
                    Item_Fat_Content: str,
                    Item_Visibility: float,
                    Item_Type:str,
                    Item_MRP: float,
                    Outlet_Establishment_Year: float,
                    Outlet_Size: str,
                    Outlet_Location_Type: str,
                    Outlet_Type: str,
                    Outlet_Identifier: str,
                 ):
        try:
            self.Item_Identifier= Item_Identifier
            self.Item_Weight = Item_Weight
            self.Item_Fat_Content = Item_Fat_Content
            self.Item_Visibility = Item_Visibility
            self.Item_Type = Item_Type
            self.Item_MRP = Item_MRP
            self.Outlet_Establishment_Year = Outlet_Establishment_Year
            self.Outlet_Size = Outlet_Size
            self.Outlet_Location_Type = Outlet_Location_Type
            self.Outlet_Type = Outlet_Type
            self.Outlet_Identifier = Outlet_Identifier
            
        except Exception as e:
            raise SalesException(e, sys) from e

    def numerical_categorical_column(self,dataframe):
        '''
        This function returns the numerical and categorical column 
        return numerical_columns,categorical_columns
        '''
        numerical_columns = [i for i in dataframe.columns if  dataframe[i].dtype != 'O']
        categorical_columns = [i for i in dataframe.columns if  dataframe[i].dtype == 'O']
        return numerical_columns,categorical_columns
    

    def encoding(self,dataframe):
        '''Item_Identifier = {'DR': 0, 'FD': 1, 'NC': 2}
        Item_Fat_Content = {'Low Fat': 0, 'Non Edible': 1, 'Regular': 2}
        Outlet_Size = {'High': 0, 'Medium': 1, 'Small': 2, np.nan: 3}
        Outlet_Location_Type = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
        Outlet_Type = {'Grocery Store': 0,
        'Supermarket Type1': 1,
        'Supermarket Type2': 2,
        'Supermarket Type3': 3}'''

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

    def get_sales_input_data_frame(self):

        try:
            sales_input_dict = self.get_sales_data_as_dict()
            df = pd.DataFrame(sales_input_dict)
            target_column_name = 'Item_Outlet_Sales'
            df = df.drop([target_column_name],axis=1)
            df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['LF','low fat','reg'],['Low Fat','Low Fat','Regular'])
            df['Item_Identifier'] = df['Item_Identifier'].apply(lambda x:x[:2])
            df['Outlet_age'] = 2013 - df['Outlet_Establishment_Year']
            df.drop(columns=['Outlet_Establishment_Year'],inplace=True)
            df.loc[df['Item_Identifier']=='NC','Item_Fat_Content'] = 'Non Edible'
            df.drop(columns=['Item_Type','Outlet_Identifier'],inplace = True)

            numerical_features,categorical_features = self.numerical_categorical_column(df)
            num_df = df[numerical_features]
            cat_df = df[categorical_features]


            cat_df = self.encoding(cat_df)
            cat_df['Outlet_Size'].replace(0,np.nan,inplace=True)

            final_df = pd.concat([num_df,cat_df],axis=1)
            
            return final_df
        except Exception as e:
            raise SalesException(e, sys) from e

    def get_sales_data_as_dict(self):
        try:
            input_data = {
                "Item_Identifier": [self.Item_Identifier],
                "Item_Weight": [self.Item_Weight],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Visibility": [self.Item_Visibility],
                "Item_Type": [self.Item_Type],
                "Item_MRP": [self.Item_MRP],
                "Outlet_Establishment_Year": [self.Outlet_Establishment_Year],
                "Outlet_Size": [self.Outlet_Size],
                "Outlet_Location_Type": [self.Outlet_Location_Type],
                "Outlet_Type" : [self.Outlet_Type],
                "Outlet_Identifier" : [self.Outlet_Identifier],
                
                }
            return input_data
        except Exception as e:
            raise SalesException(e, sys) from e


class salesPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise SalesException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            #latest_model_path = r'D:\\Projects_new\\Stores_Sales_Prediction\\saved_models\\20220728125409\\model.pkl'
            return latest_model_path
        except Exception as e:
            raise SalesException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            store_sales = model.predict(X)
            return store_sales
        except Exception as e:
            raise SalesException(e, sys) from e