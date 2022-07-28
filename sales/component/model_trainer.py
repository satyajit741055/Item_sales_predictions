import sys
from sales.exceptions import SalesException
from sales.logger import logging

from sales.entity.artifcat_entity import DataTransformationArtifact, ModelTrainerArtifact
from sales.entity.config_entity import ModelTrainerConfig
from sales.util.util import load_numpy_array_data,save_object,load_object
from sales.entity.model_factory import MetricInfoArtifact
from sales.entity.model_factory import evaluate_regression_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np


class salesEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"




class ModelTrainer:

    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SalesException(e, sys) from e

           

    def get_best_param_rf(self,x_train,y_train,x_test,y_test):
        try:
            best_params={}

            param1={'criterion': ['squared_error', 'absolute_error']}
            param2={'max_depth' : range(3,10,1)}
            param3={'max_features' : [i/100.0 for i in range(70,100,3)]}
            param4={'max_samples' : [i/100.0 for i in range(70,100,5)]}
            param5={'n_estimators':range(10,100,5)}


            parameters=[param1, param2, param3, param4, param5]

            for param in parameters:
                grid =GridSearchCV(RandomForestRegressor(), param, cv=5, n_jobs=-1)
                grid.fit(x_train, y_train)
                best_params.update(grid.best_params_)

            criterion=best_params['criterion']
            max_depth=best_params['max_depth']
            max_features=best_params['max_features']
            max_samples=best_params['max_samples']
            n_estimators=best_params['n_estimators']



            model=RandomForestRegressor(criterion=criterion, max_depth = max_depth, max_features = max_features, max_samples = max_samples, n_estimators = n_estimators)
            model.fit(x_train, y_train)
            y_pred=model.predict(x_test)

            r2=r2_score(y_test, y_pred)
            return model, r2
        except Exception as e:
            raise SalesException(e,sys) from e


    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
         
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            ### ADding New Code 
            rf_model , rf_r2 = self.get_best_param_rf(x_train=x_train,y_train=y_train,y_test=y_test,x_test=x_test)

            if rf_r2  > base_accuracy:
                best_model,model_name = rf_model  ,'RF Model '
                logging.info (f'best Model is {model_name} with parameters {best_model} ')
            else:
                raise f"None of model has base accuracy more than {base_accuracy}" 
            
            logging.info(f"Best model found on training dataset: {best_model}")

            best_model.fit( x_train,y_train)
            y_pred=best_model.predict(x_test)
            r2=r2_score(y_test, y_pred)

            logging.info(f'Model Accuracy : {r2}')

            ### ADding New Code

            preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            trained_model_file_path=self.model_trainer_config.trained_model_file_path
            metric_info:MetricInfoArtifact = evaluate_regression_model(model_list=[best_model],X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)

            model_object = metric_info.model_object 

            data_model = salesEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path,obj=data_model)
            
            model_trainer_artifact= ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
                                    trained_model_file_path=trained_model_file_path,
                                    train_rmse=metric_info.train_rmse,
                                    test_rmse=metric_info.test_rmse,
                                    train_accuracy=metric_info.train_accuracy,
                                    test_accuracy=metric_info.test_accuracy,
                                    model_accuracy=metric_info.model_accuracy
            
            )


            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SalesException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")


