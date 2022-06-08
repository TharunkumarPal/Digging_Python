#!/usr/bin/env python
# coding: utf-8

# # KNN CLASSIFICATION
# 
# **Problem Statement:**
# Prepare a model for glass classification using KNN
# 
# **Data Description:**
# 
# RI : refractive index
# 
# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 
# Mg: Magnesium
# 
# AI: Aluminum
# 
# Si: Silicon
# 
# K:Potassium
# 
# Ca: Calcium
# 
# Ba: Barium
# 
# Fe: Iron
# 
# Type: Type of glass: (class attribute)
#  
#  1 -- building_windows_float_processed
#  
#  2 --building_windows_non_float_processed
#  
#  3 --vehicle_windows_float_processed
#  
#  4 --vehicle_windows_non_float_processed (none in this database)
#  
#  5 --containers
#  
#  6 --tableware
#  
#  7 --headlamps

# In[106]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[107]:


data=pd.read_csv('glass.csv')


# In[108]:


data.head()


# In[109]:


data.info()


# In[110]:


data.describe()


# In[111]:


x=data.iloc[:,0:9]
y=data.iloc[:,9]


# In[112]:


x


# In[113]:


y.value_counts()


# In[114]:


sns.pairplot(data,hue='Type',palette='spring')


# In[115]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
StandardScaler(copy=True, with_mean=True, with_std=True)
#perform transformation
x_scaled = scaler.transform(x)


# In[116]:


x_scaled


# In[117]:


num_folds = 5
kfold = KFold(n_splits=12)


# In[118]:


model = KNeighborsClassifier(n_neighbors=9)
results = cross_val_score(model, x_scaled, y, cv=kfold)


# In[119]:


print(results.mean())


# In[120]:




# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# ## Grid Search Algorithm

# In[121]:


# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[122]:


n_neighbors = numpy.array([2*i+1 for i in range(0,5)])
param_grid = dict(n_neighbors=n_neighbors)


# In[123]:


n_neighbors


# In[124]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=9)
grid.fit(x_scaled, y)


# In[125]:


print(grid.best_score_)
print(grid.best_params_ )


# In[126]:



# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# ## Visualizing Grid Search Results

# In[127]:


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

