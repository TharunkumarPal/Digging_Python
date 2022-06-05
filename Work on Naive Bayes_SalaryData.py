#!/usr/bin/env python
# coding: utf-8

# # NAIVE BAYES
# **Problem Statement:** Prepare a classification model using Naive Bayes for salary data 
# 
# **Data Description:**
# age -- age of a person
# 
# workclass	-- A work class is a grouping of work 
# 
# education	-- Education of an individuals	
# 
# maritalstatus -- Marital status of an individulas	
# 
# occupation	 -- occupation of an individuals
# 
# relationship -- 	
# 
# race --  Race of an Individual
# 
# sex --  Gender of an Individual
# 
# capitalgain --  profit received from the sale of an investment	
# 
# capitalloss	-- A decrease in the value of a capital asset
# 
# hoursperweek -- number of hours work per week	
# 
# native -- Native of an individual
# 
# Salary -- salary of an individual
# 

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import seaborn as sns
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[5]:


data_train=pd.read_csv('Salarydata_train(1).csv')
data_test=pd.read_csv('Salarydata_test(1).csv')


# In[7]:


data_train.head()


# In[8]:


data_test.head()


# In[11]:


#Label Encoder
from sklearn.preprocessing import LabelEncoder


# In[14]:


label_encoder = preprocessing.LabelEncoder()
data_train['workclass']= label_encoder.fit_transform(data_train['workclass']) 
data_train['education']= label_encoder.fit_transform(data_train['education']) 
data_train['maritalstatus']= label_encoder.fit_transform(data_train['maritalstatus']) 
data_train['occupation']= label_encoder.fit_transform(data_train['occupation']) 
data_train['relationship']= label_encoder.fit_transform(data_train['relationship']) 
data_train['race']= label_encoder.fit_transform(data_train['race']) 
data_train['sex']= label_encoder.fit_transform(data_train['sex']) 
data_train['native']= label_encoder.fit_transform(data_train['native']) 
data_train['Salary']= label_encoder.fit_transform(data_train['Salary']) 


# In[15]:


data_train


# In[20]:


label_encoder = preprocessing.LabelEncoder()
data_test['workclass']= label_encoder.fit_transform(data_test['workclass']) 
data_test['education']= label_encoder.fit_transform(data_test['education']) 
data_test['maritalstatus']= label_encoder.fit_transform(data_test['maritalstatus']) 
data_test['occupation']= label_encoder.fit_transform(data_test['occupation']) 
data_test['relationship']= label_encoder.fit_transform(data_test['relationship']) 
data_test['race']= label_encoder.fit_transform(data_test['race']) 
data_test['sex']= label_encoder.fit_transform(data_test['sex']) 
data_test['native']= label_encoder.fit_transform(data_test['native']) 
data_test['Salary']= label_encoder.fit_transform(data_test['Salary']) 


# In[21]:


data_test.head()


# In[23]:


x_train=data_train.iloc[:,0:-1]
y_train=data_train.iloc[:,-1]


# In[24]:


x_test=data_test.iloc[:,0:-1]
y_test=data_test.iloc[:,-1]


# In[25]:


model=GaussianNB()
model=model.fit(x_train,y_train)


# In[27]:


y_pred=model.predict(x_test)


# In[28]:


metrics.accuracy_score(y_test,y_pred)

