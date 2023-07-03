#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[29]:


"""
What? Feature Importance 

The methods tested are:
[1] Linear Regression Feature Importance
[2] Logistic Regression Feature Importance
[3] CART Regression Feature Importance
[4] CART Classification Feature Importance
[5] Random Forest Regression Feature Importance
[6] Random Forest Classification Feature Importance
[7] Permutation Feature Importance for Regression
[8] Permutation Feature Importance for Classification
[9] Putting all together

"""


# In[28]:


# Import python modules
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# ### Linear Regression Feature Importance

# In[6]:


"""
These coefficients can provide the basis for a crude feature importance score. 
This assumes that the input variables have the same scale or have been scaled 
prior to fitting a model. The complete example of linear regression coefficients
for feature importance is listed below.
"""
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance) 
pyplot.show()


# In[ ]:


"""
The scores suggest that the model found the five important features and marked
all other features with a zero coefficient, essentially removing them from the model.
"""


# ### Logistic Regression Feature Importance

# In[ ]:


"""
These coefficients can provide the basis for a crude feature importance score. 
This assumes that the input variables have the same scale or have been scaled 
prior to fitting a model.
"""


# In[9]:


# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
    random_state=1)
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance): 
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[ ]:


"""
This is a classification problem with classes 0 and 1. Notice that the coefficients 
are both positive and negative. The positive scores indicate a feature that predicts 
class 1, whereas the negative scores indicate a feature that predicts class 0. No 
clear pattern of important and unimportant features can be identified from these results,
at least from what I can tell.
"""


# ### CART Regression Feature Importance

# In[11]:


# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance): 
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# ### CART Classification Feature Importance

# In[13]:


# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
    random_state=1)
# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance): 
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# ### Random Forest Regression Feature Importance

# In[15]:


# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# ### Random Forest Classification Feature Importance

# In[18]:


# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
    random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance): 
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# ### Permutation Feature Importance

# In[ ]:


"""
Permutation feature importance is a technique for calculating relative importance 
scores that is independent of the model used. First, a model is fit on the dataset,
such as a model that does not support native feature importance scores. Then the model
is used to make predictions on a dataset, although the values of a feature (column) in
the dataset are scrambled. This is repeated for each feature in the dataset. Then this 
\whole process is repeated 3, 5, 10 or more times. The result is a mean importance score
for each input feature (and distribution of scores given the repeats)
"""


# ### Permutation Feature Importance for Regression

# In[21]:


# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1) # define the model
model = KNeighborsRegressor()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance) 
pyplot.show()


# ### Permutation Feature Importance for Classification

# In[27]:


# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
    random_state=1)
# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='accuracy') # get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance) 
pyplot.show()


# ### Putting all together

# In[31]:


# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectFromModel(RandomForestClassifier(n_estimators=1000), max_features=5)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# define the dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
    random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat) 
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:


"""
In this case, we can see that the model achieves the same performance on the dataset,
although with half the number of input features. As expected, the feature importance 
scores. 
"""

