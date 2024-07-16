# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:35:43 2024

@author: rafae
"""
# File para almacenar funciones que utilizaremos en el TOG
# Para AED:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def dqr(data):
    # List of database variables
    cols = pd.DataFrame(list(data.columns.values),
                           columns=['Names'],
                           index=list(data.columns.values))
    # List of data types
    dtyp = pd.DataFrame(data.dtypes,columns=['Type'])
    # List of missing data
    misval = pd.DataFrame(data.isnull().sum(),
                                  columns=['Missing_values'])
    # List of present data
    presval = pd.DataFrame(data.count(),
                                  columns=['Present_values'])
    # List of unique values
    unival = pd.DataFrame(columns=['Unique_values'])
    # List of min values
    minval = pd.DataFrame(columns=['Min_value'])
    # List of max values
    maxval = pd.DataFrame(columns=['Max_value'])
    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except:
            pass
    
    # Join the tables and return the result
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval)