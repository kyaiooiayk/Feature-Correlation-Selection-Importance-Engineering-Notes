#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Feature-engineering" data-toc-modified-id="Feature-engineering-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Feature engineering</a></span></li><li><span><a href="#Import-modules" data-toc-modified-id="Import-modules-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Import modules</a></span></li><li><span><a href="#Polynomial-features" data-toc-modified-id="Polynomial-features-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Polynomial features</a></span></li><li><span><a href="#Feature-engineering" data-toc-modified-id="Feature-engineering-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Feature engineering</a></span></li><li><span><a href="#Sonar-dataset" data-toc-modified-id="Sonar-dataset-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Sonar dataset</a></span></li><li><span><a href="#Evaluate-ML-model-on-original-data" data-toc-modified-id="Evaluate-ML-model-on-original-data-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Evaluate ML model on original data</a></span></li><li><span><a href="#Evalute-ML-model-on-polynomial-feature-transform-example" data-toc-modified-id="Evalute-ML-model-on-polynomial-feature-transform-example-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Evalute ML model on polynomial feature transform example</a></span></li><li><span><a href="#Effect-of-polynomial-degree" data-toc-modified-id="Effect-of-polynomial-degree-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Effect of polynomial degree</a></span></li><li><span><a href="#Try-different-degrees-in-a-loop" data-toc-modified-id="Try-different-degrees-in-a-loop-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Try different degrees in a loop</a></span></li><li><span><a href="#References" data-toc-modified-id="References-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** Feature engineering - how to derive new input variables
# 
# <br></font>
# </div>

# # Feature engineering

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Often, the input features for a predictive modeling task interact in unexpected and often nonlinear ways. 
# - These interactions can be identified and modeled by a learning algorithm. 
# - Another approach is to engineer new features that expose these interactions and see if they improve model performance. 
# - Additionally, transforms like raising input variables to a power can help to better expose the important relationships between input variables and the target variable.
# 
# <br></font>
# </div>

# # Import modules

# In[1]:


from numpy import mean, std
from pandas import DataFrame
from numpy import asarray
from sklearn.preprocessing import PolynomialFeatures
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl

rcParams['figure.figsize'] = 17, 8
rcParams['font.size'] = 20
mpl.rcParams['figure.dpi']= 300


# # Polynomial features

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Polynomial features are those features created by raising existing features to an exponent. 
# - For example, if a dataset had one input feature X, then a polynomial feature would be the addition of a new feature  (column) where values were calculated by squaring the values in X, e.g. X2.
# 
# <br></font>
# </div>

# # Feature engineering

# In[2]:


# define the dataset
data = asarray([[2,3],[2,3],[2,3]])
print(data)
# perform a polynomial features transform of the dataset
trans = PolynomialFeatures(degree=2)
data = trans.fit_transform(data)
print(data)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The features created include:
# - The bias (the value of 1.0)
# - Values raised to a power for each degree (e.g. x^1, x^2, x^3, ...)
# - Interactions between all pairs of features (e.g. x^1 × x^2, x^1 × x^3, ...)
# 
# <br></font>
# </div>

# # Sonar dataset

# In[3]:


# load dataset
dataset = read_csv('../DATASETS/sonar.csv', header=None)
# summarize the shape of the dataset 
print(dataset.shape)
# summarize each variable 
dataset.describe()


# In[4]:


# histograms of the variables
fig = dataset.hist(xlabelsize = 10, ylabelsize = 10) 
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()


# # Evaluate ML model on original data

# In[5]:


# load dataset
dataset = read_csv('../DATASETS/sonar.csv', header=None)
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# define and configure the model
model = KNeighborsClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
# report model performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# # Evalute ML model on polynomial feature transform example

# In[6]:


# load dataset
dataset = read_csv('../DATASETS/sonar.csv', header=None)
# retrieve just the numeric input values
data = dataset.values[:, :-1]
print("Dataset shape BEFORE: ", dataset.shape)

# perform a polynomial features transform of the dataset 
trans = PolynomialFeatures(degree=3)
data = trans.fit_transform(data)

# convert the array back to a dataframe
dataset = DataFrame(data)
# summarize
print("Dataset shape AFTER: ", dataset.shape)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We can see that our features increased from 61 (60 input features) for the raw dataset to 39,711 features (39,710 input features).
# 
# <br></font>
# </div>

# In[7]:


# load dataset
dataset = read_csv('../DATASETS/sonar.csv', header=None)
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the pipeline
trans = PolynomialFeatures(degree=3)
model = KNeighborsClassifier()
pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# evaluate the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
# report pipeline performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Running the example, we can see that the polynomial features transform results in a lift in performance from 79.7 percent accuracy 
# - Without the transform to about 80.0 percent with the transform.
# 
# <br></font>
# </div>

# # Effect of polynomial degree

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The degree of the polynomial dramatically increases the number of input features. 
# - To get an idea of how much this impacts the number of features, we can perform the transform with a range of different degrees and compare the number of features in the dataset.
# 
# <br></font>
# </div>

# In[8]:


# get the dataset
def get_dataset(filename):
    # load dataset
    dataset = read_csv(filename, header=None)
    data = dataset.values
    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))
    return X, y

# define dataset
X, y = get_dataset('../DATASETS/sonar.csv')

# calculate change in number of features 
num_features = list()
degress = [i for i in range(1, 6)]

for d in degress:
    # create transform
    trans = PolynomialFeatures(degree=d)
    # fit and transform
    data = trans.fit_transform(X)
    # record number of features
    num_features.append(data.shape[1])

# summarize
print('Degree: %d, Features: %d' % (d, data.shape[1])) # plot degree vs number of features
pyplot.plot(degress, num_features)
pyplot.show()


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - More features may result in more overfitting, and in turn, worse results. 
# - It may be a good idea to treat the degree for the polynomial features transform as a hyperparameter and test different values for your dataset. 
# - The example below explores degree values from 1 to 4 and evaluates their effect on classification accuracy with the chosen model
# 
# <br></font>
# </div>

# # Try different degrees in a loop

# In[9]:


# get the dataset
def get_dataset(filename):
    # load dataset
    dataset = read_csv(filename, header=None)
    data = dataset.values
    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))
    return X, y

# get a list of models to evaluate
def get_models():
    models = dict()
    for d in range(1,5):
        # define the pipeline
        trans = PolynomialFeatures(degree=d)
        model = KNeighborsClassifier()
        models[str(d)] = Pipeline(steps=[('t', trans), ('m', model)])
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
    return scores


# define dataset
X, y = get_dataset('../DATASETS/sonar.csv')
# get the models to evaluate
models = get_models()

# evaluate the models and store results 

results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores))) 


# In[10]:


# plot model performance for comparison 
pyplot.boxplot(results, labels=names, showmeans=True) 
pyplot.show()


# # References

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Data preparation for machine learning, Jason Brownlee
# 
# <br></font>
# </div>

# In[ ]:




