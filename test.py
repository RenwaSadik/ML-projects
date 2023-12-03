# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:26:41 2023

@author: renwa
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report
dataset = pd.read_csv('diabetes.csv') 

print(dataset.columns)

print(dataset.isnull().sum())

print(dataset.describe())

X = dataset.iloc[:,[1,5]].values 
Y = dataset.iloc[:, -1].values


 
print(dataset['Outcome'].value_counts().plot(kind='bar'))

dataset.hist(bins=50,figsize=(20,15))
plt.show()


plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr(),annot=True)
plt.show()



from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 7) 

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=2) 
classifier.fit(X_train, y_train)

 



y_pred = classifier.predict(X_test) 
print ( 'clasification report:',classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
print(cm) 
Ac= classifier.score(X_test, y_test) 
print(Ac) 
 


 
x =dataset['BMI'] 
 
y =dataset['Glucose']
plt.scatter(x, y, c ="blue")
 
# To show the plot
plt.show()


from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) 
plt.contourf(X1, X2, 
              classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
              alpha = 0.75, cmap = 
              ListedColormap(('cyan', 'magenta'))) 
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
for i, j in enumerate(np.unique(y_set)): plt.scatter(X_set[y_set == j, 0], 
                                                      X_set[y_set == j, 1], c = ListedColormap(('blue', 'white'))(i), label = j) 
plt.title('logastic (Training set)') 
plt.xlabel('Glucose') 
plt.ylabel('BMI') 
plt.legend() 
plt.show() 

from matplotlib.colors import ListedColormap 
X_set, y_set = X_test, y_test 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                                                              stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, 
                                                                                                                    stop = X_set[:, 1].max() + 1, step = 0.01)) 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
                                                   X2.ravel()]).T).reshape(X1.shape), 
              alpha = 0.75, cmap = ListedColormap(('cyan', 'magenta'))) 
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
for i, j in enumerate(np.unique(y_set)): 
     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('blue', 'magenta'))(i), label = j) 
plt.title('logastic (Test set)') 
plt.xlabel('Glucose') 
plt.ylabel('BMI') 
plt.legend() 
plt.show() 
from sklearn import metrics 
print(" The Accuracy of the LogisticRegression :",metrics.accuracy_score(y_test, y_pred))
