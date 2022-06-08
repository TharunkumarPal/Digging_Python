#!/usr/bin/env python
# coding: utf-8

# # KNN CLASSIFICATION
# 
# **Problem Statement:**
# Implement a KNN model to classify the animals in to categorie
# 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


data=pd.read_csv('Zoo.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[8]:


data['animal name'].value_counts()


# In[9]:


data['type'].value_counts()


# In[10]:


x=data.iloc[:,1:17]
y=data.iloc[:,17]


# In[11]:


x


# In[12]:


y.value_counts()


# In[14]:


sns.pairplot(data,hue='type',palette='spring')


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
StandardScaler(copy=True, with_mean=True, with_std=True)
#perform transformation
x_scaled = scaler.transform(x)


# In[16]:


x_scaled


# In[34]:


num_folds = 5
kfold = KFold(n_splits=10)


# In[35]:


model = KNeighborsClassifier(n_neighbors=5)
results = cross_val_score(model, x_scaled, y, cv=kfold)


# In[36]:


print(results.mean())


# ## Grid Search Algorithm

# In[21]:


# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[22]:


n_neighbors = numpy.array([2*i+1 for i in range(0,5)])
param_grid = dict(n_neighbors=n_neighbors)


# In[23]:


n_neighbors


# In[24]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=9)
grid.fit(x_scaled, y)


# In[25]:


print(grid.best_score_)
print(grid.best_params_ )


# In[26]:




# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# ## Visualizing Grid Search Results

# In[27]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = [2*i+1 for i in range(0,5)]
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_scaled, y, cv=9)
    k_scores.append(scores.mean())
# plot to see clearly
plt.bar(k_range, k_scores)
plt.plot(k_range, k_scores,color="red")

plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.xticks(k_range)
plt.ylim(0.6,1)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




