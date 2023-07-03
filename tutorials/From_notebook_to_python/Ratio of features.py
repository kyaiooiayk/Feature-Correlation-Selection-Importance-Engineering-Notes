#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-the-goal-of-feature-engineering?" data-toc-modified-id="What-is-the-goal-of-feature-engineering?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is the goal of feature engineering?</a></span></li><li><span><a href="#Import-modules" data-toc-modified-id="Import-modules-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Import modules</a></span></li><li><span><a href="#Import-the-dataset" data-toc-modified-id="Import-the-dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import the dataset</a></span></li><li><span><a href="#Establish-baseline-model" data-toc-modified-id="Establish-baseline-model-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Establish baseline model</a></span></li><li><span><a href="#Feature-engineering" data-toc-modified-id="Feature-engineering-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Feature engineering</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** FE - introduction ratio of features
# 
# <br></font>
# </div>

# # What is the goal of feature engineering?

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - You might perform feature engineering to:
#     - improve a model's predictive performance
#     - reduce computational or data needs
#     - improve interpretability of the results
# 
# <br></font>
# </div>

# # Import modules

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# # Import the dataset

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - To illustrate these ideas we'll see how adding a few synthetic features to a dataset can improve the predictive performance of a random forest model.
# - The Concrete dataset contains a variety of concrete formulations and the resulting product's *compressive strength*, which is a measure of how much load that kind of concrete can bear. 
# - The task for this dataset is to predict a concrete's compressive strength given its formulation.
# 
# <br></font>
# </div>

# In[2]:


df = pd.read_excel("../DATASETS/Concrete_Data.xls")
df.head()


# # Establish baseline model

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We'll first establish a baseline by training the model. 
# - This will help us determine whether our new features are actually useful. 
# 
# <br></font>
# </div>

# In[3]:


X = df.copy()
y = X.pop("CompressiveStrength")

# Train and score baseline model
baseline = RandomForestRegressor(criterion="mae", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")


# # Feature engineering

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The *ratio* of ingredients in a recipe is usually a better predictor of how the recipe turns out than their absolute amounts. 
# - We might reason then that ratios of the features above would be a good predictor of `CompressiveStrength`.
# - The cell below adds three new ratio features to the dataset. 
# 
# <br></font>
# </div>

# In[4]:


X = df.copy()
y = X.pop("CompressiveStrength")

# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score model on dataset with additional ratio features
model = RandomForestRegressor(criterion="mae", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - And sure enough, performance improved! 
# - This is evidence that these new ratio features exposed important information to the model that it wasn't detecting before.
# 
# <br></font>
# </div>

# # References

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# - https://www.kaggle.com/ryanholbrook/what-is-feature-engineering
# - https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength 
# 
# <br></font>
# </div>

# In[ ]:




