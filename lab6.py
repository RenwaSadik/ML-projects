# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:09:19 2023

@author: renwa
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
 
# the dataset 
dataset = pd.read_csv('Social_Network_Ads(3).csv') 
X = dataset.iloc[:, [2, 3]].values 
y = dataset.iloc[:, -1].values 

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0) 
 
#scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 


from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators=12, criterion='entropy') 
classifier.fit(X_train, y_train) 


y_pred = classifier.predict(X_test) 


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print(cm) 
Ac= classifier.score(X_test, y_test) 
print(Ac) 
 
 
 
# Visualising 
from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) 
plt.contourf(X1, X2, 
              classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
              alpha = 0.75, cmap = 
              ListedColormap(('red', 'green'))) 
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
for i, j in enumerate(np.unique(y_set)): plt.scatter(X_set[y_set == j, 0], 
                                                      X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j) 
plt.title('RandomForestClassifier (Training set)') 
plt.xlabel('Age') 
plt.ylabel('Estimated Salary') 
plt.legend() 
plt.show() 

from matplotlib.colors import ListedColormap 
X_set, y_set = X_test, y_test 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                                                              stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, 
                                                                                                                    stop = X_set[:, 1].max() + 1, step = 0.01)) 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
                                                   X2.ravel()]).T).reshape(X1.shape), 
              alpha = 0.75, cmap = ListedColormap(('red', 'green'))) 
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
for i, j in enumerate(np.unique(y_set)): 
     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j) 
plt.title('RandomForestClassifier (Test set)') 
plt.xlabel('Age') 
plt.ylabel('Estimated Salary') 
plt.legend() 
plt.show() 
from sklearn import metrics 
print(" The Accuracy of the RandomForestClassifier :",metrics.accuracy_score(y_test, y_pred))