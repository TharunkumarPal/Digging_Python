#!/usr/bin/env python
# coding: utf-8

# # CLUSTERING

# **Problem Statement:** 
# To perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.
# 
# **Data:** Crime data
#     
# **Data Description:**
# Murder -- Muder rates in different places of United States
# Assualt- Assualt rate in different places of United States
# UrbanPop - urban population in different places of United States
# Rape - Rape rate in different places of United States
# 
# **Methodolgy:**
#     In this particular work we tried clustering of the given using 
#     
#       1. Hierarchial Clustering
#       2. Kmeans
#       3. DBScan
#     
# Optimum no of clusters are developed based on the various techniques and are clearly mentioned in the below.

# # KMeans

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")


# In[2]:


crime=pd.read_csv('crime_data.csv')
crime.head()


# In[3]:


scaler = MinMaxScaler()
crime_scaled = scaler.fit_transform(crime.iloc[:,1:5])


# In[4]:


crime_scaled


# In[5]:


model = KMeans(n_clusters=4, max_iter=600, algorithm = 'auto',init="k-means++")
model.fit(crime_scaled)


# In[6]:


model.fit(crime_scaled)


# In[7]:


pred=model.predict(crime_scaled)
pred


# In[8]:


model.cluster_centers_


# In[9]:


#Scatterplot for urbanpop vs Rape
plt.scatter(crime.iloc[:,3],crime.iloc[:,4],c=pred,cmap=plt.cm.autumn) 


# In[10]:


for i in range(2,10):
    model1 = KMeans(n_clusters=i, max_iter=600, algorithm = 'auto',init="k-means++",)
    model1.fit(crime_scaled)    
    pred=model1.predict(crime_scaled)    
    
    plt.scatter(crime.iloc[:,2],crime.iloc[:,3],c=pred,cmap=plt.cm.spring)
    plt.title(str(i)+ " clusters ")
    plt.show()


# In[11]:


wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, max_iter=600, algorithm = 'auto',init="k-means++",)
    model.fit(crime_scaled)    
    wcss.append(model.inertia_) #inertia is another name for wcss
    
plt.plot(range(1, 11), wcss,color='orange')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# **Conclusion:** From the Elbow method, optimal values of clusters is found to be 4

# 
# # Hierarchial Clustering

# In[12]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# In[13]:


crime.head()


# In[14]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[15]:


# Normalized data frame (considering the numerical part of data)
crime_norm = norm_func(crime.iloc[:,1:])


# In[16]:


crime_norm.iloc[:,:]


# In[17]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(crime_norm, method='average'))
# create clusters
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'average')


# In[18]:


hc


# In[19]:


hc.fit_predict(crime_norm)


# In[20]:


# save clusters for chart
y_hc = hc.fit_predict(crime_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[21]:


crime['h_clusterid'] = y_hc 
y_hc


# In[22]:


import matplotlib.pyplot as plt
plt.boxplot(crime["Rape"])
plt.show()


# In[23]:


crime


# In[24]:


crime.groupby(["h_clusterid"])["Unnamed: 0"].apply(lambda x:tuple(x))


# ## DBSCAN 

# In[37]:


from sklearn.cluster import DBSCAN


# In[26]:


crime_scaled


# In[27]:


model_db=DBSCAN(eps=0.2,min_samples=3)
model_db.fit(crime_scaled)


# In[28]:


model_db.labels_


# In[29]:


df2=pd.DataFrame(model_db.labels_,columns=['cluster'])


# In[30]:


df2.head()


# In[31]:


crime["cluster_dbscan"]=df2["cluster"]


# In[32]:


crime["cluster_dbscan"].value_counts()


# In[33]:


crime.groupby(['cluster_dbscan']).agg(['mean']).reset_index()


# In[34]:


crime.groupby(["cluster_dbscan"])["Unnamed: 0"].apply(lambda x:tuple(x))


# In[35]:


import matplotlib.pyplot as plt
plt.style.use('classic')


# In[36]:


crime.plot(x="Murder",y ="Assault",c= model.labels_ ,kind="scatter",s=50,cmap=plt.cm.plasma ) 
plt.title('Clusters using DBScan')  

