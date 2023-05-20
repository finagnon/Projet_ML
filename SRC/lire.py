import random
import tqdm
import matplotlib.pyplot as plt
from more_itertools import chunked
import numpy as np
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class ARIMAModel:
    def __init__(self, p, d, q, t_train, t_validation, t_test, y_train, y_validation, y_test):
        self.p = p
        self.d = d
        self.q = q
        self.t_train = t_train
        self.t_validation = t_validation
        self.t_test = t_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test

    def training(self):
        self.arima = ARIMA(self.y_train, order=(self.p, self.d, self.q)).fit()
        print(self.arima.summary())
        with open("summary.txt","w") as sumary:
            sumary.write(str(self.arima.summary()))
    def show_forcast_of_arima_model(self):
        if not hasattr(self, "arima"):
            print("Erreurl: Il faudra faire le train d'abord!!")
        else:
            train_predictions = self.arima.predict(start=0, end=len(self.t_train) + len(self.t_validation) + len(self.t_test) - 1, typ='levels')
            validation_predictions = self.arima.predict(start=len(self.t_train), end=len(self.t_train) + len(self.t_validation) - 1, typ='levels')
            test_predictions = self.arima.predict(start=len(self.t_train) + len(self.t_validation), end=len(self.t_train) + len(self.t_validation) + len(self.t_test) - 1, typ='levels')
            rmse = np.sqrt(mean_squared_error(self.y_test, test_predictions))

            
            print("Début mesure RMSE".center(50,"-"))
            
            print("RMSE:", rmse)
           
            print("Fin mesure RMSE".center(50,"-"))
            
             

            plt.figure(figsize=(15, 6))
            # train base
            plt.plot(self.t_train, self.y_train, "o", color='b', label="Train base")
            # validation base
            plt.plot(self.t_validation, self.y_validation, "o", color='r', label="Validation base")
            # test base
            plt.plot(self.t_test, self.y_test, "o", color='g', label="Test base")

            plt.plot(np.concatenate([self.t_train, self.t_validation, self.t_test]), train_predictions, "-", color='b', label="Train prediction")
            plt.plot(self.t_validation, validation_predictions, "-", color='r', label="Validation prediction")
            plt.plot(self.t_test, test_predictions, "-", color='g', label="Test prediction")

            plt.legend()
            plt.show();
            