#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" 
What? Feature selection.

Feature selection is a process where you automatically select 
those features in your data that contribute most to the prediction 
variable or output in which you are interested. Having irrelevant 
features in your data can decrease the accuracy of many models,
especially linear algorithms like linear and logistic regression. 

Three benefits are:
    [1] Reduces Overfitting
    [2] Improves Accuracy
    [3] Reduces Training Time
   
"""


# In[1]:


# Import python modules
from pandas import read_csv
from numpy import set_printoptions
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from IPython.display import Markdown, display
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


# Additional functions
def myPrint(string, c = "blue"):    
    """My version of the python-native print command.
    
    Print in bold and red tect
    """
    colorstr = "<span style='color:{}'>{}</span>".format(c, '**'+ string + '**' )    
    display(Markdown(colorstr))

def printPythonModuleVersion():    
    """printPythonModuleVersion
    Quickly list the python module versions
    """
    myPrint("Checking main python modules version")
    import scipy
    print('scipy: %s' % scipy.__version__)
    import numpy
    print('numpy: %s' % numpy.__version__)    
    import matplotlib
    print('matplotlib: %s' % matplotlib.__version__)    
    import pandas
    print('pandas: %s' % pandas.__version__)
    import statsmodels
    print('statsmodels: %s' % statsmodels.__version__) 
    import sklearn
    print('sklearn: %s' % sklearn.__version__)
    import xgboost
    print('xgboostn: %s' % xgboost.__version__)    

printPythonModuleVersion()


# In[3]:


# Read-in the data
filename = './datasetCollections/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
myPrint("Checking size of input and labels")
print("Input's shape: ", X.shape)
print("label's shape: ", Y.shape)
myPrint("Input names")
print(dataframe.columns)
myPrint("Print dataframe")
print(dataframe)


# In[24]:


# [1] UNIVARIATE SELECTION

"""
Statistical tests can be used to select those features 
that have the strongest relationship with the output variable. 
The scikit-learn library provides the SelectKBest class2 
that can be used with a suite of different statistical tests
to select a specific number of features. The example below uses 
the chi-squared (chi2) statistical test for non-negative features 
to select 4 of the best features
"""

# feature extraction
test = SelectKBest(score_func = chi2, k = 4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)

print(fit.scores_)

features = fit.transform(X)
# summarize selected features
print(features[0:5,:])


# In[60]:


# [2] RECURSIVE FEATURE ELIMINATION

"""
The Recursive Feature Elimination (or RFE) works by recursively removing
attributes and building a model on those attributes that remain.
The example below uses RFE with the logistic regression algorithm to 
select the top 3 features.
"""

# Feature extraction
model = LogisticRegression(max_iter = 500)
rfe = RFE(model, n_features_to_select = 3)
fit = rfe.fit(X, Y)

myPrint("No of features selected")
print(fit.n_features_)
myPrint("Complete map")
for i in range(8):
    print("Feature: ", dataframe.columns[i], " selected? ", fit.support_[i], " rank? ", fit.ranking_[i])


# In[66]:


# [3] # Feature Extraction with PCA

"""
Principal Component Analysis (or PCA) uses linear algebra to transform the 
dataset into a compressed form. Generally this is called a data reduction
technique. A property of PCA is that you can choose the number of dimensions
or principal components in the transformed result. In the example below, we 
use PCA and select 3 principal components.

8 :: No of features
"""

# feature extraction
pca = PCA(n_components = 3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s", fit.explained_variance_ratio_)
print(fit.components_)
print(fit.components_.shape)
print(X.shape)


# In[72]:


# [4] # Feature Importance with Extra Trees Classifier

"""
Bagged decision trees like Random Forest and Extra Trees 
can be used to estimate the importance of features.
"""

# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

myPrint("Complete map")
a = 0
for i in range(8):
    print("Feature: -->>", dataframe.columns[i], "<<-- level of importance? ", model.feature_importances_[i])
    a+=model.feature_importances_[i]
    print("Cumulative importance value: ", a)


# In[ ]:




