# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('DataLab1.csv')


x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 4].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')
imputer = imputer.fit(x[:, 1:4])
x[:, 1:4] = imputer.transform(x[:, 1:4])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

x = np.array(ct.fit_transform(x), dtype= float)

y = LabelEncoder().fit_transform(y)

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3 , random_state =0 ) 



from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)