# -*- coding: utf-8 -*-
"""
Created on Sat May 13 13:49:27 2023

@author: fiacr
"""
# importation de bibliothèque
from prepro import preprocessing
from prepare import prepare_data
from lire import ARIMAModel

def main():
    
    df = preprocessing("temperature.csv")
    t_train, t_test ,t_validation, y_train, y_test, y_validation=prepare_data(df, 0.6, 0.2)
    
    print(t_train)
    print(t_test)
    print(t_validation)
    
    print(y_train)
    print(y_test)
    print(y_validation)
    p = ARIMAModel(12,1,12,t_train, t_test ,t_validation, y_train, y_test, y_validation)
    p.training()
    p.show_forcast_of_arima_model()
    
    
    
if __name__=="__main__": 
        
        main()
        
        
    