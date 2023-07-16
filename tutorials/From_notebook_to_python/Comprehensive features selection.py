#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Import-modules" data-toc-modified-id="Import-modules-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import modules</a></span></li><li><span><a href="#UDF" data-toc-modified-id="UDF-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>UDF</a></span></li><li><span><a href="#Import-dataset" data-toc-modified-id="Import-dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import dataset</a></span></li><li><span><a href="#Splitting-dataset" data-toc-modified-id="Splitting-dataset-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Splitting dataset</a></span></li><li><span><a href="#Relationship-between-the-random-feature-and-the-target" data-toc-modified-id="Relationship-between-the-random-feature-and-the-target-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Relationship between the random feature and the target</a></span></li><li><span><a href="#Benchmark-model-[aka-baseline]" data-toc-modified-id="Benchmark-model-[aka-baseline]-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Benchmark model [aka baseline]</a></span></li><li><span><a href="#Default-Scikit-learn's-feature-importances" data-toc-modified-id="Default-Scikit-learn's-feature-importances-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Default Scikit-learn's feature importances</a></span></li><li><span><a href="#Permutation-feature-importance" data-toc-modified-id="Permutation-feature-importance-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Permutation feature importance</a></span><ul class="toc-item"><li><span><a href="#rfpimp" data-toc-modified-id="rfpimp-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>rfpimp</a></span></li><li><span><a href="#eli5" data-toc-modified-id="eli5-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>eli5</a></span></li></ul></li><li><span><a href="#Drop-Column-feature-importance" data-toc-modified-id="Drop-Column-feature-importance-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Drop Column feature importance</a></span></li><li><span><a href="#Observation-level-feature-importance" data-toc-modified-id="Observation-level-feature-importance-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Observation level feature importance</a></span><ul class="toc-item"><li><span><a href="#treeinterpreter" data-toc-modified-id="treeinterpreter-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>treeinterpreter</a></span></li><li><span><a href="#LIME" data-toc-modified-id="LIME-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>LIME</a></span></li></ul></li></ul></div>

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** Understaning feature importance<br>
# 
# **Reference:** https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e<br>
# **Reference:** https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/feature_importance.ipynb<br>
# 
# <br></font>
# </div>

# # Import modules

# In[1]:


import eli5
import lime
import numpy as np
import pandas as pd
import seaborn as sns
import lime.lime_tabular
from pylab import rcParams
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
#from rfpimp import plot_corr_heatmap
from sklearn.metrics import r2_score
#from rfpimp import permutation_importances
from eli5.sklearn import PermutationImportance
from sklearn.base import clone
from treeinterpreter import treeinterpreter as ti, utils


# # UDF

# In[2]:


def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
       .set_title(title, fontsize = 20)


# # Import dataset

# In[3]:


"""
For this example, I will use the Boston house prices dataset (so a regression problem).
"""


# In[4]:


boston = load_boston()
print(boston.DESCR)


# In[5]:


y = boston.target
X = pd.DataFrame(boston.data, columns = boston.feature_names)
X.head(5)


# In[6]:


"""
Why adding a random column?
The only non-standard thing in preparing the data is the addition of a random column to the dataset. 
Logically, it has no predictive power over the dependent variable (Median value of owner-occupied homes 
in $1000's), so it should not be an important feature in the model. Let’s see how it will turn out.
"""


# In[7]:


np.random.seed(seed = 42)
X['random'] = np.random.random(size = len(X))
X.head(5)


# # Splitting dataset

# In[8]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.8, random_state = 42)


# # Relationship between the random feature and the target 

# In[9]:


"""
As it can be observed, there is no pattern on the scatterplot and the correlation is almost 0.
"""


# In[10]:


rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 20
plt.scatter(X['random'],y)


# In[11]:


rcParams['figure.figsize'] = 18, 10
rcParams['font.size'] = 20

sns.heatmap(X.assign(target = y).corr().round(2), cmap = 'Blues', 
            annot = True).set_title('Correlation matrix', fontsize = 5)


# In[12]:


"""
The DIFFERENCE between standard Pearson’s correlation is that this one first transforms variables into ranks and 
only then runs Pearson’s correlation on the ranks.

Spearman’s correlation:
    - is nonparametric
    - does not assume a linear relationship between variables
    - it looks for monotonic relationships
"""


# In[13]:


#from rfpimp import plot_corr_heatmap
#viz = plot_corr_heatmap(X_train, figsize=(18,10))
#viz.view()


# # Benchmark model [aka baseline]

# In[14]:


"""
random_state =42 -> to ensure results comparability
oob_score = True -> so I could later use the out-of-bag error.

There is some overfitting in the model, as it performs much worse on OOB sample and worse on the validation set.
"""


# In[15]:


# Instantiate the model
get_ipython().run_line_magic('time', '')
rf = RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)
# Fit the model
rf.fit(X_train, y_train) 


# In[16]:


print("R^2 Training Score: ", rf.score(X_train, y_train))
print("OOB Score: ", rf.oob_score_)
print("R^2 Validation Score: ", rf.score(X_valid, y_valid))


# # Default Scikit-learn's feature importances

# In[17]:


"""
In decision trees, every node is a condition how to split values in a single feature, so that similar values of dependent variable end up in the same set after the split. 

The condition is based on impurity, which in case of: 
    - Classification problems is Gini impurity / information gain (entropy), 
    - Regression trees its variance

Thus, when training a tree we can compute how much each feature contributes to decreasing the weighted impurity. 

feature_importances_ in Scikit-Learn (for Random Forest) averages the decrease in impurity over trees.

Pros:
    - fast calculation
    - easy to retrieve - one command
Cons:
     - biased approach, as it has a tendency to inflate the importance of continuous features or high-cardinality 
       categorical variables
"""


# In[18]:


base_imp = imp_df(X_train.columns, rf.feature_importances_)
base_imp


# In[19]:


var_imp_plot(base_imp, 'Default feature importance (scikit-learn)')


# In[20]:


"""
What seems surprising though is that a column of random values turned out to be more than other variables!
"""


# # Permutation feature importance

# In[21]:


"""
This approach directly measures feature importance by observing how random re-shuffling (thus preserving the
distribution of the variable) of each predictor influences model performance.

Pros:
    - applicable to any model
    - reasonably efficient
    - reliable technique
    - no need to retrain the model at each modification of the dataset

Cons:
    - more computationally expensive than default feature_importances
    - permutation importance overestimates the importance of correlated predictors 
"""


# ## rfpimp

# In[23]:


def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)
perm_imp_rfpimp.reset_index(drop = False, inplace = True)


# In[ ]:


var_imp_plot(perm_imp_rfpimp, 'Permutation feature importance (rfpimp)')


# In[ ]:


"""
The plot confirms what we have seen above, that 4 variables are less important than a random variable! 
Surprising... The top 4 stayed the same though. One more nice feature about rfpimp is that it contains 
functionalities for dealing with the issue of collinear features (that was the idea behind showing the 
Spearman's correlation matrix). 
"""


# ## eli5

# In[24]:


perm = PermutationImportance(rf, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)


# In[25]:


var_imp_plot(perm_imp_eli5, 'Permutation feature importance (eli5)')


# In[ ]:


"""
The results are very similar to the previous ones, even as these came from multiple reshuffles per column.
The default importance DataFrame is not the most readable, as it does not contain variable names. This can
be of course quite easily fixed. The nice thing is the standard error from all iterations of the reshuffling 
on each variable.
"""


# In[26]:


eli5.show_weights(perm)


# # Drop Column feature importance

# In[27]:


"""
This approach is quite an intuitive one, as we investigate the importance of a feature by comparing a model 
with all features versus a model with this feature dropped for training.
"""


# In[28]:


def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    
    importances_df = imp_df(X_train.columns, importances)
    return importances_df


# In[29]:


drop_imp = drop_col_feat_imp(rf, X_train, y_train)


# In[30]:


imp_df.columns = ['feature', 'feature_importance']
sns.barplot(x = 'feature_importance', y = 'feature', 
data = drop_imp, orient = 'h', color = 'royalblue') \
   .set_title('Drop column feature importance', fontsize = 20)


# In[31]:


"""
Negative importance in this case means that removing a given feature from the model actually improves the 
performance. So this is nice to see in case of random, but what is weird is that the highest performance boost 
can be observed after removing DIS, which was the third most important variable in previous approaches.
WHY IS THIS HAPPENNING? COMBINATION HAS AN EFFECT?
"""


# # Observation level feature importance

# In[32]:


"""
For the observation with the SMALLEST error, the main contributor was LSTAT and RM (which in previous cases 
turned out to be most important variables). 
In the highest error case, the HIGHEST contribution came from DIS variable, overcoming the same two variables
that played the most important role in the first case.

We'll use two libraries:
    - treeinterpreter
    - LIME
    
An discussion on the treeinterpreter vs LIME can be found here:
https://stackoverflow.com/questions/48909418/lime-vs-treeinterpreter-for-interpreting-decision-tree/48975492
"""


# ## treeinterpreter

# In[33]:


selected_rows = [31, 85]
selected_df = X_train.iloc[selected_rows,:].values
prediction, bias, contributions = ti.predict(rf, selected_df)

for i in range(len(selected_rows)):
    print("Row", selected_rows[i])
    print("Prediction:", prediction[i][0], 'Actual Value:', y_train[selected_rows[i]])
    print("Bias (trainset mean)", bias[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions[i], 
                                 X_train.columns), 
                             key=lambda x: -abs(x[0])):
        print(feature, round(c, 2))
    print("-"*20) 


# In[34]:


"""
To dive even deeper, we might also be interested in joined contribution of many variables
"""


# In[35]:


prediction1, bias1, contributions1 = ti.predict(rf, np.array([selected_df[0]]), joint_contribution=True)
prediction2, bias2, contributions2 = ti.predict(rf, np.array([selected_df[1]]), joint_contribution=True)


# In[36]:


aggregated_contributions1 = utils.aggregated_contribution(contributions1)
aggregated_contributions2 = utils.aggregated_contribution(contributions2)


# In[37]:


res = []
for k in set(aggregated_contributions1.keys()).union(
              set(aggregated_contributions2.keys())):
    res.append(([X_train.columns[index] for index in k] , 
               aggregated_contributions1.get(k, 0) - aggregated_contributions2.get(k, 0)))   
         
for lst, v in (sorted(res, key=lambda x:-abs(x[1])))[:10]:
    print (lst, v)


# In[38]:


"""
The most of the difference between the best and worst predicted cases comes from the number of rooms (RM) feature, 
in conjunction with weighted distances to five Boston employment centres (DIS).
"""


# ## LIME

# In[39]:


"""
LIME (Local Interpretable Model-agnostic Explanations) is a technique explaining the predictions of any 
classifier/regressor in an interpretable and faithful manner. To do so, an explanation is obtained by 
locally approximating the selected model with an interpretable one (such as linear models with regularisation or decision trees). The interpretable models are trained on small perturbations (adding noise) of the original observation (row in case of tabular data), thus they only provide a good local approximation.

Some drawbacks to be aware of:
    - only linear models are used to approximate local behavior
    - type of perturbations that need to be performed on the data to obtain correct explanations are often 
      use-case specific
    - simple (default) perturbations are often not enough. In an ideal case, the modifications would be driven
      by the variation that is observed in the dataset
"""


# In[40]:


explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   mode = 'regression',
                                                   feature_names = X_train.columns,
                                                   categorical_features = [3], 
                                                   categorical_names = ['CHAS'], 
                                                   discretize_continuous = True)
                                                   
np.random.seed(42)
exp = explainer.explain_instance(X_train.values[31], rf.predict, num_features = 5)
exp.show_in_notebook(show_all=False) #only the features used in the explanation are displayed

exp = explainer.explain_instance(X_train.values[85], rf.predict, num_features = 5)
exp.show_in_notebook(show_all=False)


# In[41]:


"""
LIME interpretation agrees that for these two observations the most important features are RM and LSTAT, 
which was also indicated by previous approaches.

"""


# In[ ]:





# In[ ]:





# In[ ]:




