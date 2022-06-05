#!/usr/bin/env python
# coding: utf-8

# # CLUSTERING

# **Problem Statement:** Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters. 
# Draw the inferences from the clusters obtained.
# 
# **Data Description:**
#  
# The file EastWestAirlinescontains information on passengers who belong to an airline’s frequent flier program. For each passenger the data include information on their mileage history and on different ways they accrued or spent miles in the last year. The goal is to try to identify clusters of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers
# 
# ID --Unique ID
# 
# Balance--Number of miles eligible for award travel
# 
# Qual_mile--Number of miles counted as qualifying for Topflight status
# 
# cc1_miles -- Number of miles earned with freq. flyer credit card in the past 12 months:
# cc2_miles -- Number of miles earned with Rewards credit card in the past 12 months:
# cc3_miles -- Number of miles earned with Small Business credit card in the past 12 months:
# 
# 1 = under 5,000
# 2 = 5,000 - 10,000
# 3 = 10,001 - 25,000
# 4 = 25,001 - 50,000
# 5 = over 50,000
# 
# Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months
# 
# Bonus_trans--Number of non-flight bonus transactions in the past 12 months
# 
# Flight_miles_12mo--Number of flight miles in the past 12 months
# 
# Flight_trans_12--Number of flight transactions in the past 12 months
# 
# Days_since_enrolled--Number of days since enrolled in flier program
# 
# Award--whether that person had award flight (free flight) or not
# 
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

# In[28]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")


# In[29]:


airlines=pd.read_csv('EastWestAirlines.csv')
airlines.head()


# In[30]:


airlines.info()


# In[31]:


#standardizing thr data
scaler = MinMaxScaler()
airlines_scaled = scaler.fit_transform(airlines.iloc[:,1:13])
airlines_scaled


# In[32]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[33]:


# Normalized data frame (considering the numerical part of data)
airlines_norm = norm_func(airlines.iloc[:,1:])
airlines_norm.iloc[:,:]


# In[34]:


model_kmeans = KMeans(n_clusters=4, max_iter=600, algorithm = 'auto',init="k-means++")
model_kmeans.fit(airlines_scaled)


# In[35]:


pred=model_kmeans.predict(airlines_scaled)
pred


# In[36]:


model_kmeans.cluster_centers_


# In[37]:


#Scatterplot for Balance vs Bonusmiles
plt.figure(figsize=(10,8))
plt.scatter(airlines.iloc[:,1],airlines.iloc[:,10],c=pred,cmap=plt.cm.viridis)
plt.xlabel=('Bonus miles')
plt.ylabel('Balance')


# In[38]:


# For visualizations

for i in range(2,10):
    model = KMeans(n_clusters=i, max_iter=600, algorithm = 'auto',init="k-means++",)
    model.fit(airlines_scaled)    
    pred=model.predict(airlines_scaled)    
    
    plt.scatter(airlines.iloc[:,10],airlines.iloc[:,1],c=pred,cmap=plt.cm.viridis)
    plt.title(str(i)+ " clusters ")
    plt.show()


# In[39]:


wcss = []
for i in range(1, 15):
    model = KMeans(n_clusters=i, max_iter=600, algorithm = 'auto',init="k-means++",)
    model.fit(airlines_scaled)    
    wcss.append(model.inertia_) #inertia is another name for wcss
    
plt.plot(range(1, 15), wcss,color='red')
plt.title('Elbow Method')
plt.ylabel('WCSS')
plt.show()


# **Note:** From Elbow method the optimim value of k is 3 or 4

# In[40]:


airlines['Clus_KMeans']=model_kmeans.labels_


# In[41]:


airlines


# In[42]:


airlines.groupby('Clus_KMeans').agg(['mean']).reset_index()


# ## Hierarchial Clustering

# In[43]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# In[44]:


airlines_norm


# In[51]:


# create dendrogram
plt.figure(figsize=(15,10))
dendrogram = sch.dendrogram(sch.linkage(airlines_norm, method='average'))
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'complete')


# In[46]:


hc.fit(airlines_norm)


# In[56]:


# save clusters for chart
cluster_hc = hc.fit_predict(airlines_norm)
clusters=pd.DataFrame(cluster_hc,columns=['Clusters'])


# In[57]:


clusters


# In[64]:


airlines["Clus_HC"]=clusters


# In[65]:


airlines


# In[70]:


airlines.groupby(["Clus_HC"])["ID#"].apply(lambda x:tuple(x))


# # DBSCAN

# In[72]:


from sklearn.cluster import DBSCAN


# In[73]:


airlines_scaled


# In[84]:


model_db=DBSCAN(eps=0.35,min_samples=3)
model_db.fit(airlines_scaled)


# In[85]:


model_db.labels_


# In[86]:


df2=pd.DataFrame(model_db.labels_,columns=['Clus_DB'])


# In[87]:


df2


# In[88]:


df2.value_counts()


# In[90]:


airlines["Clus_dbscan"]=df2["Clus_DB"]


# In[91]:


airlines


# In[92]:


airlines.groupby(['Clus_dbscan']).agg(['mean']).reset_index()


# In[119]:


import matplotlib.pyplot as plt
plt.style.use('dark_background')


# In[120]:


airlines.plot(x="Balance",y ="Bonus_miles",c= model_db.labels_ ,kind="scatter",s=50,cmap=plt.cm.plasma ) 
plt.title('Clusters using DBScan')  

