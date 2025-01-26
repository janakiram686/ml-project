import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def evaluate_model(self,X_train,y_train,X_test,y_test,models):
        try:
            report={}

            for i in range(len(list(models))):
                model=list(models.values())[i]

                model.fit(X_train,y_train)

                y_train_pred=model.predict(X_train)

                y_test_pred=model.predict(X_test)

                train_model_score=r2_score(y_train,y_train_pred)

                test_model_score=r2_score(y_test,y_test_pred)

                report[list(models.keys())[i]]=test_model_score

            return report

        except:
            pass



    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split training and test input data")

            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "random forest":RandomForestRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "gradient boost": GradientBoostingRegressor(),
                "linear regression":LinearRegression(),
                "k_neighbors" :KNeighborsRegressor(),
                "xgb regressor":XGBRegressor(),
                "cat boost regressor":CatBoostRegressor(),
                "ada boost regressor":AdaBoostRegressor(),
            }

            model_report:dict =self.evaluate_model(X_train,y_train,X_test,y_test,models)

            ## best model from dict
            best_model_score=max(model_report.values())

            ## best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if  best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best found model on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted=best_model.predict(X_test)

            r2score=r2_score(y_test,predicted)
            return r2score,best_model_name
            
        except Exception as e:
            raise CustomException(e,sys)
        