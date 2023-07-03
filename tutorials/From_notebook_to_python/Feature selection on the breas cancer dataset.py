#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Feature selection on the breas cancer dataset
# 
# </font>
# </div>

# In[ ]:


"""
What? Breast cancer categorical dataset

The so-called Breast cancer dataset that has been widely studied as a machine learning dataset since the 1980s.
The dataset classifies breast cancer patient data as either a recurrence or no recurrence of cancer. There are 286
examples and 9 input variables. It is a binary classification problem. A naive model can achieve an accuracy of 
70 percent on this dataset. A good score is about 76 percent.
"""


# # Import modules

# In[64]:


from random import seed
from pandas import read_csv
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif,chi2, mutual_info_classif,SelectPercentile,SelectFpr,SelectFdr, SelectFwe,GenericUnivariateSelect
from IPython.display import Markdown, display
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
 
seed(1)


# # Plan of action

# In[ ]:


"""
As a first step, we will evaluate a LogisticRegression model using ALL the available features. Logistic regression 
is a good model for testing feature selection methods as it can perform better if irrelevant features are removed 
from the model. To recap:

[1] Use all the feature
[2] Remove some feature
[3] Compare predictions
"""


# # Load dataset

# In[9]:


def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    # format all fields as string
    X = X.astype(str)
    return X, y


# In[10]:


X, y = load_dataset('../DATASETS/breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) 


# # Prepare input data

# In[11]:


def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# In[12]:


# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)


# # Prepare target

# In[13]:


def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


# In[14]:


y_train_enc, y_test_enc = prepare_targets(y_train, y_test)


# # Fit the model with -->>ALL<<-- the features

# In[15]:


# Instantiate the model first
model = LogisticRegression(solver = 'lbfgs')
# Perform actual fitting
model.fit(X_train_enc, y_train_enc)


# # Evaluate the model

# In[ ]:


"""
In this case, we can see that the model achieves a classification accuracy of about 75 percent. We would prefer to
use a subset of features that achieves a classification accuracy that is as good or better than this.
"""


# In[16]:


yhat = model.predict(X_test_enc)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))


# # Feature selection

# In[ ]:


"""
Under the hood the function ask you to use a score function and the number of feature you want to select. 
In this case we use chi2 with k=4 hence 4 features out of the 8 features available.
"""


# In[45]:


def select_features(X_train, y_train, X_test, scoreMethod):
    print("Selected score_func:", scoreMethod)
    fs = SelectKBest(score_func = scoreMethod, k = 4)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs


# In[47]:


X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc, chi2)


# # Fit the model with -->>SOME<<-- the features

# In[48]:


# Instantiate the modela again 
model = LogisticRegression(solver = 'lbfgs')
model.fit(X_train_fs, y_train_enc)


# # Evaluate the model

# In[ ]:


"""
In this case, we see that the model achieved an accuracy of about 74 percent, a slight drop in performance. 
It is possible that some of the features removed are, in fact, adding value directly or in concert with the 
selected features. At this stage, we would probably prefer to use all of the input features.
"""


# In[50]:


yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))


# # Feature selection using different score function

# In[28]:


"""

f_classif: ANOVA F-value between label/feature for classification tasks.mutual_info_classif Mutual 
information for a discrete target.

chi2: Chi-squared stats of non-negative features for classification tasks. 

f_regression: F-value between label/feature for regression tasks.

mutual_info_regression: Mutual information for a continuous target.

SelectPercentile: Select features based on percentile of the highest scores.

SelectFpr: Select features based on a false positive rate test.

SelectFdr: Select features based on an estimated false discovery rate.

SelectFwe: Select features based on family-wise error rate.

GenericUnivariateSelect: Univariate feature selector with configurable mode.
"""


# In[85]:


def runInLoop(scoreMethod):
    # feature selection
    X_train_fs, X_test_fs = select_features(
    X_train_enc, y_train_enc, X_test_enc, scoreMethod)
    # fit the model
    #model1 = LogisticRegression(solver = 'lbfgs')
    model1 = LogisticRegression()
    model1.fit(X_train_fs, y_train_enc)
    # evaluate the model
    yhat = model1.predict(X_test_fs)
    # evaluate predictions
    accuracy = accuracy_score(y_test_enc, yhat)
    print('Accuracy: %.2f' % (accuracy*100))


# In[93]:


scoreMethodsList = [f_classif,chi2, mutual_info_classif]
for singleScoreMethod in scoreMethodsList:
    runInLoop(singleScoreMethod)


# In this case, we can see a small lift in classification accuracy to 76 percent. To be sure that the effect is real, it would be a good idea to repeat each experiment multiple times and compare the mean performance. It may also be a good idea to explore using k-fold cross-validation instead of a simple train/test split.

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
# 
# </font>
# </div>

# In[ ]:




