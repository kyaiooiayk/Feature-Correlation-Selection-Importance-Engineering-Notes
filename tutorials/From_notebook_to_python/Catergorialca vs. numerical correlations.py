#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Create-synthetic-dataset" data-toc-modified-id="Create-synthetic-dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Create synthetic dataset</a></span></li><li><span><a href="#Point-biserial-correlation" data-toc-modified-id="Point-biserial-correlation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Point-biserial correlation</a></span></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Catergorialca vs. numerical correlations
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[6]:


from matplotlib import pyplot
from scipy.stats import pointbiserialr
import pandas as pd


# # Create synthetic dataset
# <hr style="border:2px solid black"> </hr>

# In[11]:


df = pd.DataFrame()
df["cat"] = [1,1,1,1,1,2,2,2,2,22] 
df["num"] = [1,2,3,4,5,6,7,8,10,11]


# In[8]:


df


# # Point-biserial correlation
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - A point-biserial correlation is simply the correlation between one dichotmous variable and one continuous variable.
# - It turns out that this is a special case of the Pearson correlation. So computing the special point-biserial correlation is equivalent to computing the Pearson correlation when one variable is dichotmous and the other is continuous.
#     
# </font>
# </div>

# In[13]:


pointbiserialr(df["cat"], df["num"]).correlation


# In[14]:


pointbiserialr(df["cat"], df["num"]).pvalue


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [Point-biserial correlation, Phi, & Cramer's V](https://web.pdx.edu/~newsomj/pa551/lectur15.htm)
#     
# </font>
# </div>

# In[ ]:




