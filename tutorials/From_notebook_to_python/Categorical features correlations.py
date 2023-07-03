#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-the-issue-with-correlation-btw-categoricals?" data-toc-modified-id="What-is-the-issue-with-correlation-btw-categoricals?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is the issue with correlation btw categoricals?</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-dataset" data-toc-modified-id="Load-dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Load dataset</a></span></li><li><span><a href="#One-hot-encoding" data-toc-modified-id="One-hot-encoding-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>One-hot encoding</a></span></li><li><span><a href="#Cramer's-V" data-toc-modified-id="Cramer's-V-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Cramer's V</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Categorical features correlation
# 
# </font>
# </div>

# # What is the issue with correlation btw categoricals?
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Correlation is a measure of the association between two variables. 
# - It is easy to calculate and interpret when both variables are numerical, but about the case where they are all categoricals and correlation isn’t defined in that case. How do you subtract yellow from the average of colors? 
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[1]:


from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
from numpy import cov
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib import rcParams


# # Load dataset
# <hr style="border:2px solid black"> </hr>

# In[2]:


df = pd.read_csv("./mushrooms.csv")


# In[3]:


df


# In[4]:


df.isna().any().sum()


# # One-hot encoding
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - One common option to handle this scenario is by first using one-hot encoding, and break each possible option of each categorical feature to 0-or-1 features. This will then allow the use of correlation, but it can easily become too complex to analyse. For example, one-hot encoding converts the 22 categorical features of the mushrooms data-set to a 112-features data-set
# - This is not something that can be easily used for gaining new insights. So we still need something else.
#     
# </font>
# </div>

# In[5]:


df_oh = pd.get_dummies(df)


# In[6]:


correlation_mat = df_oh.corr("pearson")


# In[7]:


correlation_mat


# In[8]:


mask = np.zeros_like(correlation_mat, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
correlation_mat[mask] = np.nan


# In[9]:


rcParams["figure.figsize"] = 14,10
rcParams["font.size"] = 10

sns.heatmap(correlation_mat, annot = True)
plt.show()


# # Cramer's V
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - It is used to evaluate the association btw two categorical features where the ontigency matrix is larger than 2x2, for instance a 2x3 matrix. In this case Phi is not appropriate.
# - It is based on a nominal variation of Pearson’s Chi-Square Test, and comes built-in with some great benefits:
# - Similarly to correlation, the output is in the range of [0,1], where 0 means no association and 1 is full association. 
# - Unlike correlation, there are no negative values, as there’s no such thing as a negative association. Either there is, or there isn’t.
# - Like correlation, Cramer’s V is symmetrical — it is insensitive to swapping x and y.
#     
# </font>
# </div>

# In[10]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[11]:


cramers_v(df["class"], df["cap-shape"])


# In[12]:


df.shape


# In[13]:


df.head(3)


# In[29]:


dummy = np.zeros([23, 23])
for i, i_value in enumerate(df.columns):
    for j, j_value in enumerate(df.columns):
        dummy[i,j] = cramers_v(df[i_value], df[j_value])


# In[30]:


dummy


# In[31]:


mask = np.zeros_like(dummy, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
dummy[mask] = np.nan


# In[32]:


dummy


# In[33]:


dummy = pd.DataFrame(dummy, columns=df.columns)


# In[34]:


dummy = dummy.rename(index={i:j for i,j in enumerate(df.columns)})


# In[35]:


{str(i):j for i,j in enumerate(df.columns)}


# In[36]:


dummy


# In[37]:


rcParams["figure.figsize"] = 14,10
rcParams["font.size"] = 10

sns.heatmap(dummy, annot = True)
plt.show()


# <div class="alert alert-info">
# <font color=black>
# 
# - Just by looking at this heat-map we can see that the odor is highly associated with the class (edible/poisonous) of the mushroom, and that the gill-attachment feature is highly associated with three others.
#     
# </font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [The search for categorical correlation](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)
# - [Mushroom classificaiton](https://www.kaggle.com/datasets/uciml/mushroom-classification)
# - [Point-biserial correlation, Phi, & Cramer's V](https://web.pdx.edu/~newsomj/pa551/lectur15.htm)
#     
# </font>
# </div>
