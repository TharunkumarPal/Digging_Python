#!/usr/bin/env python
# coding: utf-8

# # PRINCIPAL COMPONENT ANALYSIS 
# 
# **Problem Statement:**
# 
# Perform Principal component analysis and perform clustering using first 3 principal component scores (both heirarchial and k mean clustering(scree plot or elbow curve))
# 
# Check wwhether the optimum number of clusters is equal to the number of clusters given in actual data.
# 

# In[39]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# In[4]:


data=pd.read_csv('wine.csv')
data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[10]:


data['Type'].value_counts()


# In[18]:


data_tem=data.drop(data['Type'])


# In[19]:


data_tem


# In[20]:


# Normalizing the numerical data 
data_normal = scale(data_tem)


# In[21]:


data_normal


# In[22]:


pca = PCA()
pca_values = pca.fit_transform(data_normal)


# In[23]:


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var


# In[24]:


plt.bar(range(1,len(var)+1),var)


# In[25]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[26]:


pca.components_


# In[27]:


# Variance plot for PCA components obtained 
plt.plot(var1,color="red")
plt.ylim(0,100)


# In[28]:


pca_values.shape


# In[33]:


# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
plt.scatter(x,y)


# In[34]:


# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,2]
plt.scatter(x,y)


# In[35]:


data_pc=pd.DataFrame(pca_values[:,0:3], columns=['pc1','pc2','pc3'])


# In[36]:


data_pc


# ## KMEANS 
# 
# Clustering of 3 Principal Components using KMEANS

# In[96]:


model = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto',init="k-means++")
model.fit(data_pc)


# In[97]:


pred=model.predict(data_pc)
pred=pd.DataFrame(data=pred)
pred


# In[98]:


pred.value_counts()


# In[99]:


model.cluster_centers_


# In[100]:


plt.scatter(data_pc.iloc[:,0],data_pc.iloc[:,1],c=pred,cmap=plt.cm.autumn) 


# In[59]:


wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, max_iter=600, algorithm = 'auto',init="k-means++",)
    model.fit(data_pc)    
    wcss.append(model.inertia_) #inertia is another name for wcss
    
plt.plot(range(1, 11), wcss,color='orange')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# **Note:** From Elbow curve it can be observed that optimum number of clusters is 3

# ## Hierarchial Clustering
# 
# Performing Hierarchial Clustering on the 3 Principal Components

# In[60]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# In[61]:


data_pc


# In[65]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data_pc, method='complete'))
# create clusters
hc = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'complete')


# In[88]:


pred_hc=hc.fit_predict(data_pc)
pred_hc


# In[89]:


pred_hc=pd.DataFrame(data= pred_hc)


# In[101]:


pred_hc.head()


# In[102]:


pred_hc.value_counts()


# In[112]:


plt.scatter(x=data_pc.iloc[:,0],y=data_pc.iloc[:,1],c=pred_hc.iloc[:,0],cmap=plt.cm.autumn)


# In[ ]:




