#!/usr/bin/env python
# coding: utf-8

# # DECISION TREES
# 
# **Problem Statement:**
# A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
# 
# Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  
# 
# 
# About the data: 
# Let’s consider a Company dataset with around 10 variables and 400 records.
# 
# **The attributes are as follows:** 
# 
#  Sales -- Unit sales (in thousands) at each location
# 
#  Competitor Price -- Price charged by competitor at each location
# 
#  Income -- Community income level (in thousands of dollars)
# 
#  Advertising -- Local advertising budget for company at each location (in thousands of dollars)
# 
#  Population -- Population size in region (in thousands)
# 
#  Price -- Price company charges for car seats at each site
# 
#  Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
# 
#  Age -- Average age of the local population
# 
#  Education -- Education level at each location
# 
#  Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
# 
#  US -- A factor with levels No and Yes to indicate whether the store is in the US or not
#  
# 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import metrics


# In[4]:


data=pd.read_csv("Company_Data.csv")


# In[12]:


data.head()


# In[14]:


data[data.duplicated()]


# In[6]:


y=data['Sales']
Y=pd.DataFrame(y)
Y.describe()


# In[11]:


plt.figure(figsize=(20,5))
sns.boxplot(Y['Sales'],color= "plum")


# In[23]:


cat=[]
for i in data['Sales']:
    if i<= 5.39:
        cat.append('low')
        
    elif (i>5.39) & (i<=9.32):
        cat.append('moderate')
    else: 
        cat.append('high')


# In[24]:


Y.describe()


# In[25]:


cat


# In[26]:


Y['cat']=pd.DataFrame(cat)
Y


# In[27]:


data['Class']=Y['cat']
data


# In[28]:


data['Class'].value_counts()


# In[29]:


label_encoder = preprocessing.LabelEncoder()
data['ShelveLoc']= label_encoder.fit_transform(data['ShelveLoc']) 
data['Urban']= label_encoder.fit_transform(data['Urban']) 
data['US']= label_encoder.fit_transform(data['US']) 
data['Class']= label_encoder.fit_transform(data['Class']) 


# In[30]:


data


# In[31]:


sns.pairplot(data)


# In[32]:


data.head()


# In[33]:


x=data.iloc[:,1:11]
tar=data.iloc[:,11]


# In[34]:


tar.head()


# In[35]:


# Splitting data into training and testing data set
x_train, x_test,tar_train,tar_test = train_test_split(x,tar, test_size=0.15)


# In[36]:


x_train


# In[42]:


tar_train.value_counts()


# In[61]:


tar_test.value_counts()


# In[66]:


model=DecisionTreeClassifier(criterion='entropy',min_samples_split=5,max_depth=6)
model.fit(x_train,tar_train)


# In[67]:


fn=['CompPrice','Income','Advertising','Population', 'Price', 'ShelveLoc','Age', 'Education','Urban','US']
cn=['high', 'low', 'moderate']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,3), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[68]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[69]:


from sklearn import metrics
metrics.accuracy_score(preds,tar_test)

