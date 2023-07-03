#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Features selection for REGRESSION problem

https://machinelearningmastery.com/feature-selection-for-regression-data/
"""


# #  Import python modules

# In[6]:


from pylab import rcParams
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from IPython.display import Markdown, display
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, mutual_info_regression


# # Checking python module versions

# In[7]:


def printPythonModuleVersion():    
    
    """printPythonModuleVersion
    Quickly list the python module versions
    """
    
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


# # Feature selection function

# In[8]:


def select_features(X_train, y_train, X_test, chosenMethod):
    # configure to select all features
    fs = SelectKBest(score_func = chosenMethod, k = 'all') 
    # learn relationship from training data 
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# # Create fictitious REGRESSION dataset

# In[ ]:


"""
The make regression() function from the scikit-learn library can be used to define a dataset.
It provides control over the number of samples, number of input features, and, importantly, 
the number of relevant and irrelevant input features.
"""


# In[9]:


# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# ### f_regression

# In[10]:


# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_regression)
# what are scores for the features
for i in range(len(fs.scores_)): 
    print('Feature %d: %f' % (i, fs.scores_[i]))


# In[11]:


# plot the scores
rcParams['figure.figsize'] = 15, 6
rcParams['font.size'] = 16
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()


# In[ ]:


"""
A bar chart of the feature importance scores for each input feature is created. 
The plot clearly shows 8 to 10 features are a lot more important than the other
features. We could set k = 10 When configuring the SelectKBest to select these 
top features.
"""


# ### mutual_info_regression

# In[ ]:


"""
Mutual information from the field of information theory is the application of 
information gain (typically used in the construction of decision trees) to 
feature selection. Mutual information is calculated between two variables and
measures the reduction in uncertainty for one variable given a known value of
the other variable. Mutual information is straightforward when considering the
distribution of two discrete (categorical or ordinal) variables, such as categorical
input and categorical output data. Nevertheless, it can be adapted for use with 
numerical input and output data.
"""


# In[12]:


# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_regression)
# what are scores for the features
for i in range(len(fs.scores_)): 
    print('Feature %d: %f' % (i, fs.scores_[i]))


# In[13]:


# plot the scores
rcParams['figure.figsize'] = 15, 6
rcParams['font.size'] = 16
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()


# In[ ]:


"""
A bar chart of the feature importance scores for each input feature is created. Compared to the correlation 
feature selection method we can clearly see many more features scored as being relevant. This may be because
of the statistical noise that we added to the dataset in its construction.
"""


# # Plan of actions

# In[ ]:


"""
There are many different techniques for scoring features and selecting features based on scores; how do you know
which one to use? A robust approach is to evaluate models using different feature selection methods (and numbers
of features) and select the method that results in a model with the best performance. In this section, we will 
evaluate a Linear Regression model with all features compared to a model built from features selected by 
correlation statistics and those features selected via mutual information. Linear regression is a good model 
for testing feature selection methods as it can perform better if irrelevant features are removed from the model.
"""


# # Model built using ALL features

# In[23]:


# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat) 
print('MAE: %.3f' % mae)


# # Modifying the select_feature function

# In[15]:


# feature selection
def select_features(X_train, y_train, X_test, chosenModel, featuresNo):
    # configure to select a subset (ONLY 10) of features
    fs = SelectKBest(score_func = chosenModel, k = featuresNo)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# # Model built using correlation features

# In[16]:


# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_regression, 10)
# fit the model
model = LinearRegression() 
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# In[ ]:


"""
In this case, we see that the model achieved an error score of about 2.7, which is 
much larger than the baseline model that used all features and achieved an MAE of 
0.086. This suggests that although the method has a strong idea of what features 
to select, building a model from these features alone does not result in a more 
skillful model. This could be because features that are important to the target 
are being left out, meaning that the method is being deceived about what is important.


-->>POSSIBLE SOLUTION<<--
Let’s go the other way and try to use the method to remove some irrelevant features 
rather than all irrelevant features. We can do this by setting the number of selected
features to a much larger value, in this case, 88, hoping it can find and discard 12 
of the 90 irrelevant features. The complete example is listed below
"""


# In[32]:


# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_regression, 88)
# fit the model
model = LinearRegression() 
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat) 
print('MAE: %.3f' % mae)


# In[ ]:


"""
In this case, we can see that removing some of the irrelevant features has resulted in 
a small lift in performance with an error of about 0.085 compared to the baseline that 
achieved an error of about 0.086.
"""


# ### Model built using mutual information features

# In[33]:


# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_regression, 88)
# fit the model
model = LinearRegression() 
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat) 
print('MAE: %.3f' % mae)


# In[ ]:


"""
In this case, we can see a further reduction in error as compared to the correlation statistic, 
achieving a MAE of about 0.084 compared to 0.085 in the previous section.
"""


# ### Tune the number of selected features

# In[37]:


# define dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,random_state=1)
# define the evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])

# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)] 

# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv) 
# perform the search
results = search.fit(X, y)
# summarize best
print('Best MAE: %.3f' % results.best_score_) 
print('Best Config: %s' % results.best_params_) 
# summarize all
means = results.cv_results_['mean_test_score'] 
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print('>%.3f with: %r' % (mean, param))


# In[ ]:


"""
In this case, we can see that the best number of selected features is 81, which achieves a MAE of about 0.082 
"""

