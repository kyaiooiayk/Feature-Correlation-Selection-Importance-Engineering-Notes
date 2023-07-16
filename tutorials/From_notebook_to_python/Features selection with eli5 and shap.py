#!/usr/bin/env python
# coding: utf-8

# # Introduction to this python notebook

# In[ ]:


""" 
What? Understaning feature importance

The most important goal of this notebook is to show the extra feature of two libraries:
[1] eli5
[2] shap

Reference: https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
           https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/feature_importance.ipynb
           https://www.kdnuggets.com/2021/01/ultimate-scikit-learn-machine-learning-cheatsheet.html
"""


# # Import python modules

# In[ ]:


import numpy as np
import pandas as pd
from pylab import rcParams
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from rfpimp import plot_corr_heatmap
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.base import clone
from treeinterpreter import treeinterpreter as ti, utils
import lime
import lime.lime_tabular
# Getting rid of the warning messages
import warnings
warnings.filterwarnings("ignore")


# # UDF

# In[ ]:


def imp_df(column_names, importances):
    
    """
    Creating a pandas dataframe in ascending order
    """
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

# In[ ]:


"""
For this example, I will use the Boston house prices dataset (so a regression problem).
"""


# In[ ]:


boston = load_boston()
print(boston.DESCR)


# In[ ]:


y = boston.target
X = pd.DataFrame(boston.data, columns = boston.feature_names)
X.head(5)


# In[ ]:


print(y)


# # Splitting dataset

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.8, random_state = 42)


# # Benchmark model [aka baseline]

# In[ ]:


"""
random_state =42 -> to ensure results comparability
oob_score = True -> so I could later use the out-of-bag error.

There is some overfitting in the model, as it performs much worse on OOB sample and worse on the validation set.
"""


# In[ ]:


# Instantiate the model
get_ipython().run_line_magic('time', '')
rf = RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)
# Fit the model
rf.fit(X_train, y_train) 


# In[ ]:


print("R^2 Training Score: ", rf.score(X_train, y_train))
print("OOB Score: ", rf.oob_score_)
print("R^2 Validation Score: ", rf.score(X_valid, y_valid))


# # Features importance vs. PCA

# In[ ]:


"""
Feature Importance is the process of finding the most important feature to a target. 

Through PCA, the feature that contains the most information can be found, but feature importance concerns a 
feature’s impact on the target. A change in an ‘important’ feature will have a large effect on the y-variable, 
whereas a change in an ‘unimportant’ feature will have little to no effect on the y-variable.
"""


# # Default Scikit-learn's feature importances

# In[ ]:


"""
In decision trees, every node is a condition how to split values in a single feature, so that similar values of
dependent variable end up in the same set after the split. 

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


# In[ ]:


base_imp = imp_df(X_train.columns, rf.feature_importances_)
base_imp


# In[ ]:


rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 17
var_imp_plot(base_imp, 'Default feature importance (scikit-learn)')


# # Permutation feature importance

# In[ ]:


"""
This approach directly measures feature importance by observing how random re-shuffling (thus preserving the
distribution of the variable) of each predictor influences model performance.

In other words, it is a method to evaluate how important a feature is. Several models are trained, each missing
one column. The corresponding decrease in model accuracy as a result of the lack of data represents how important
the column is to a model’s predictive power

Pros:
    - applicable to any model
    - reasonably efficient
    - reliable technique
    - no need to retrain the model at each modification of the dataset

Cons:
    - more computationally expensive than default feature_importances
    - permutation importance overestimates the importance of correlated predictors 
"""


# # -->>rfpimp<<-- library based on permutation

# In[ ]:


"""
One more nice feature about rfpimp is that it contains functionalities for dealing with the issue of 
collinear features (that was the idea behind the Spearman's correlation matrix). 
"""


# In[ ]:


def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)
perm_imp_rfpimp.reset_index(drop = False, inplace = True)


# In[ ]:


rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 17
var_imp_plot(perm_imp_rfpimp, 'Permutation feature importance (rfpimp)')


# # -->>eli5<<-- library based on permutation

# In[ ]:


perm = PermutationImportance(rf, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)


# In[ ]:


rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 17
var_imp_plot(perm_imp_eli5, 'Permutation feature importance (eli5)')


# In[ ]:


"""
The results are very similar to the previous ones, even as these came from multiple reshuffles per column.
The default importance DataFrame is not the most readable, as it does not contain variable names. This can
be of course quite easily fixed. The nice thing is the standard error from all iterations of the reshuffling 
on each variable.
"""


# In[ ]:


eli5.show_weights(perm, feature_names = X.columns.tolist())


# # -->>shap<<-- 

# In[ ]:


"""
SHAP is another method of evaluating feature importance, borrowing from game theory principles in Blackjack 
to estimate how much value a player can contribute. Unlike permutation importance, SHapley Addative 
ExPlanations use a more formulaic and calculation-based method towards evaluating feature importance. 
SHAP requires a tree-based model (Decision Tree, Random Forest) and accommodates both regression and 
classification.
"""


# In[ ]:


import shap
rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 17
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type = "bar")


# # -->>PDP<<-- Partial Dependence Plots

# In[ ]:


"""
Partial dependence plots show how certain values of one feature influence a change in the target variable.
Shaded areas represent confidence intervals. The change in the target value is seen on y-coordinate
"""


# In[ ]:


from pdpbox import pdp, info_plots


for singleFeatureName in X.columns.tolist():
    pdp_dist = pdp.pdp_isolate(model=rf, 
                               dataset=X, 
                               model_features=X.columns.tolist(),
                               feature=singleFeatureName)
    pdp.pdp_plot(pdp_dist, singleFeatureName)
    rcParams['figure.figsize'] = 15, 3
    rcParams['font.size'] = 17
    plt.show()


# # Contour PDPs

# In[ ]:


"""
Partial dependence plots can also take the form of contour plots, which compare not one isolated variable but 
the relationship between two isolated variables.
Ref: https://github.com/SauceCat/PDPbox/issues/40
"""


# In[ ]:


print(X.columns.tolist())


compared_features = ['CRIM', 'ZN']
inter = pdp.pdp_interact(model=rf, 
                          dataset=X, 
                          model_features=X.columns, 
                          features=compared_features)
pdp.pdp_interact_plot(pdp_interact_out=inter,
                      feature_names=compared_features,
                      plot_type='contour')
plt.show()


# # Drop Column feature importance

# In[ ]:


"""
This approach is quite an intuitive one, as we investigate the importance of a feature by comparing a model 
with all features versus a model with this feature dropped for training.
"""


# In[ ]:


def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    print("Benchmark score [baseline]: ", benchmark_score)
    # list for storing feature importances
    deltaScore = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        print("Dropping feature name: ", col)
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        #print(drop_col_score)
        #print(benchmark_score - drop_col_score)
        deltaScore.append(benchmark_score - drop_col_score)        
    
    return deltaScore


# In[ ]:


deltaScore = drop_col_feat_imp(rf, X_train, y_train)


# In[ ]:


for i, feature in enumerate(X_train.columns):
    print(feature, " has delta score of: ", deltaScore[i])    


# In[ ]:


# Create a pandas dataframe which helps to sort via ascending order
df2 = pd.DataFrame({'feature': X_train.columns,
                   'deltaScore': deltaScore}) \
       .sort_values('deltaScore', ascending = False) \
       .reset_index(drop = True)
    
rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 17


#df2.columns = ['feature', 'deltaScore']
sns.barplot(x = 'deltaScore', y = 'feature', data = df2, orient = 'h', color = 'royalblue')


# In[ ]:


"""
Negative importance in this case means that removing a given feature from the model actually improves the 
performance. What is weird is that the highest performance boost can be observed after removing DIS, which 
was the third most important variable in previous approaches.
"""


# # Observation level feature importance

# In[ ]:


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

# In[ ]:


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


# In[ ]:


"""
To dive even deeper, we might also be interested in joined contribution of many variables
"""


# In[ ]:


prediction1, bias1, contributions1 = ti.predict(rf, np.array([selected_df[0]]), joint_contribution=True)
prediction2, bias2, contributions2 = ti.predict(rf, np.array([selected_df[1]]), joint_contribution=True)


# In[ ]:


aggregated_contributions1 = utils.aggregated_contribution(contributions1)
aggregated_contributions2 = utils.aggregated_contribution(contributions2)


# In[ ]:


res = []
for k in set(aggregated_contributions1.keys()).union(
              set(aggregated_contributions2.keys())):
    res.append(([X_train.columns[index] for index in k] , 
               aggregated_contributions1.get(k, 0) - aggregated_contributions2.get(k, 0)))   
         
for lst, v in (sorted(res, key=lambda x:-abs(x[1])))[:10]:
    print (lst, v)


# In[ ]:


"""
The most of the difference between the best and worst predicted cases comes from the number of rooms (RM) feature, 
in conjunction with weighted distances to five Boston employment centres (DIS).
"""


# ## LIME

# In[ ]:


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


# In[ ]:


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


# In[ ]:


"""
LIME interpretation agrees that for these two observations the most important features are RM and LSTAT, 
which was also indicated by previous approaches.

"""

