# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:28:45 2023

@author: renwa
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import the dataset
dataset = pd.read_csv('Restaurant_Reviews_Dataset(1).tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]','',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ''.join(review)
    corpus.append(review)

#Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1550)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# Traninig the Naive Bayes model on the Training set

from sklearn.svm import SVC 
classifier = SVC(kernel='rbf',random_state=0) 
classifier.fit(X_train, y_train)

#Predicting the Test set result
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



