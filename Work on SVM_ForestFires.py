#!/usr/bin/env python
# coding: utf-8

# # SUPPORT VECTOR MACHINE
# 
# **Problem Statement:**
# classify the Size_Categorie using SVM
# 
# **Data terms:**
# 
# month	month of the year: 'jan' to 'dec'
# 
# day	day of the week: 'mon' to 'sun'
# 
# FFMC	FFMC index from the FWI system: 18.7 to 96.20
# 
# DMC	DMC index from the FWI system: 1.1 to 291.3
# 
# DC	DC index from the FWI system: 7.9 to 860.6
# 
# ISI	ISI index from the FWI system: 0.0 to 56.10
# 
# temp	temperature in Celsius degrees: 2.2 to 33.30
# 
# RH	relative humidity in %: 15.0 to 100
# 
# wind	wind speed in km/h: 0.40 to 9.40
# 
# rain	outside rain in mm/m2 : 0.0 to 6.4
# 
# Size_Categorie 	the burned area of the forest ( Small , Large)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


data=pd.read_csv('forestfires.csv')
data


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data['size_category'].value_counts()


# In[6]:


sb.countplot(data=data,x='size_category')


# In[7]:


from sklearn.preprocessing import LabelEncoder
label_encoder = preprocessing.LabelEncoder()
data['month'] = label_encoder.fit_transform(data['month'])
data['day'] = label_encoder.fit_transform(data['day'])
data['size_category'] = label_encoder.fit_transform(data['size_category'])
data


# In[8]:


x=data.iloc[:,0:11]
y=data.iloc[:,30]


# In[9]:


x


# In[10]:


y


# In[11]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[12]:


xtrain


# In[13]:


ytrain


# In[14]:


ytest.value_counts()


# In[25]:


clf = SVC()
param_grid = [{'kernel':['linear'],'gamma':[0.5,0.1] }]
gsv = GridSearchCV(clf,param_grid,cv=15)
gsv.fit(xtrain,ytrain)


# In[26]:


gsv.best_params_ , gsv.best_score_ 


# In[27]:


clf = SVC( kernel ="linear" )
clf.fit(xtrain , ytrain)
ypred = clf.predict(xtest)
acc = accuracy_score(ytest, ypred) * 100
print("Accuracy =", acc)
confusion_matrix(ytest, ypred)


# In[28]:


import warnings
warnings.filterwarnings("ignore")
print(classification_report(ytest, ypred))


# In[29]:


ypred


# In[ ]:





# In[ ]:




