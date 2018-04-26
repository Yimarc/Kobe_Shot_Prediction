%matplotlib inline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load data

# training data
train = pd.read_csv('data.csv', header=None)
y_train = train.values[:,10]
y_train[y_train == 'g'] = 0
y_train[y_train == 'h'] = 1
y_train = y_train.astype(float)
X_train = train.values[:,:10]

# testing data
test = pd.read_csv('data.csv', header=None)
y_test = test.values[:,10]
y_test[y_test == 'g'] = 0
y_test[y_test == 'h'] = 1
y_test = y_test.astype(float)
X_test = test.values[:,:10]

# standardize the data
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)