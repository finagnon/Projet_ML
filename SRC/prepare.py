# -*- coding: utf-8 -*-
"""
Created on Sat May 13 23:05:31 2023

@author: fiacr
"""

def prepare_data(df, coef_t,coef_tst):
    
    tail =len(df)
    tail_t = int(tail*coef_t) 
    tail_tst = int(tail*coef_tst)
    
    t_train = df.index[ : tail_t ]
    
    t_test  = df.index[tail_t : (tail_t+ tail_tst)]
    
    t_validation = df.index[(tail_t+ tail_tst) : ]
    
    y_train =df['temperature'][: tail_t]
    
    y_test  = df.temperature[tail_t : (tail_t+ tail_tst)]
    
    y_validation = df.temperature[(tail_t+ tail_tst) : ]
    
    return t_train, t_test ,t_validation, y_train, y_test, y_validation