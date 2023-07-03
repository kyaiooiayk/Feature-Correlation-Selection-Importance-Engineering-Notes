#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-mutual-information?" data-toc-modified-id="What-is-mutual-information?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is mutual information?</a></span></li><li><span><a href="#Practical-suggestions" data-toc-modified-id="Practical-suggestions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Practical suggestions</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Import-dataset" data-toc-modified-id="Import-dataset-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Import dataset</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Mutual information
# 
# </font>
# </div>

# # What is mutual information?
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-info">
# <font color=black>
#     
# - **Mutual information** is a similar to correlation in but it can detect **any** kind of relationship, while correlation only detects *linear* relationships.
# - The **mutual information** (MI) between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other.
# - Uncertainty is measured using "entropy". The entropy of a variable means roughly: "how many yes-or-no questions you would need to describe an occurence of that variable, on average." The more questions you have to ask, the more uncertain you must be about the variable. Mutual information is how many questions you expect the feature to answer about the target.
# - **MI=0**: quantities are independent.
# - **MI>0**: there's no upper bound to what MI can be. In practice though values above 2.0 or so are uncommon. Mutual information is a logarithmic quantity, so it increases very slowly.
# 
# </font>
# </div>

# # Practical suggestions
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Here are some things to remember when applying mutual information
#     - MI can help you to understand the *relative potential* of a feature as a predictor of the target, considered by itself.
#     - It's possible for a feature to be very informative when interacting with other features, but not so informative all alone. MI *can't detect interactions* between features. It is a **univariate** metric.
#     - The *actual* usefulness of a feature *depends on the model you use it with*. 
#     - A feature is only useful to the extent that its relationship with the target is one your model can learn. 
#     - Just because a feature has a high MI score doesn't mean your model will be able to do anything with that information. 
#     - You may need to transform the feature first to expose the association.
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression


# # Import dataset
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - The  dataset consists of 193 cars from the 1985 model year. 
# - The goal for this dataset is to predict a car's `price` (the target) from 23 of the car's features, such as `make`, `body_style`, and `horsepower`. 
# - In this example, we'll rank the features with mutual information and investigate the results by data visualization.
# - **WARNING** This particular dataset has some **?** characters. We'll first turn them into a NaN and then we'll drop the row.
# 
# </font>
# </div>

# In[2]:


df = pd.read_csv("./Automobile_data.csv")
df = df.replace("?", np.NaN)
df = df.dropna(how = 'any', axis = 0) 
df.drop(columns = "normalized-losses", inplace = True)
df.head()


# In[3]:


df["price"] = df["price"].astype("float")

df["price"].describe()


# In[4]:


df["length"].describe()


# In[5]:


df.info()


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - For label encoding we are going to use label encoding.
# - There are differne method available but pandas has a method called 'factorize()'
# - This method **will not** add an additional column.
# - See in the references for a description on how this method differs from the others.
# 
# </font>
# </div>

# In[20]:


# Before
df["fuel-type"]


# In[19]:


# After using fatorize() method
df["fuel-type"].factorize()


# In[ ]:


X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int


# In[21]:


X


# In[10]:


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(
        X, y.values.ravel(), discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


mi_scores = make_mi_scores(X, y, discrete_features)
# Show a few features with their MI scores
mi_scores[::3]


# In[11]:


# Bar plot to make comparisions easier:
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - As we might expect, the high-scoring 'curb_weight' feature exhibits a strong relationship with price, the target.
# - The `fuel_type` feature has a fairly low MI score, but as we can see from the figure, it clearly separates two `price` populations with different trends within the `horsepower` feature. 
# - This indicates that `fuel_type` contributes an interaction effect and might not be unimportant after all. Before deciding a feature is unimportant from its MI score, it's good to investigate any possible interaction effects -- domain knowledge can offer a lot of guidance here.
# 
# </font>
# </div>

# In[12]:


sns.relplot(x="curb-weight", y="price", data=df);


# In[13]:


# If you do not turn them into float, it'll throw you an error. See referece below.
df["horsepower"] = df["horsepower"].astype("float")


# In[14]:


sns.lmplot(x="horsepower", y="price", hue="fuel-type", data=df);


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# - https://www.kaggle.com/ryanholbrook/mutual-information
# - https://www.kaggle.com/toramky/automobile-dataset
# - [Why do need to cast some column while using lmplot](https://www.py4u.net/discuss/159827)
# - [pandas factorize()](https://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.factorize.html)
# - ['factorize' vs. 'LabelEncoder' vs. 'get_dummies' vs. 'OneHotEncoder'](https://stackoverflow.com/questions/40336502/want-to-know-the-diff-among-pd-factorize-pd-get-dummies-sklearn-preprocessing/40338956)
# - [Information theory](https://en.wikipedia.org/wiki/Information_theory)
#     
# </font>
# </div>

# In[ ]:




