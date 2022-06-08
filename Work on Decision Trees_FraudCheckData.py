#!/usr/bin/env python
# coding: utf-8

# # DECISION TREES
# 
# **Problem Statement:** 
# Use decision trees to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
# 
# **Data Description :**
# 
# Undergrad : person is under graduated or not
# 
# Marital.Status : marital status of a person
# 
# Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
# 
# Work Experience : Work experience of an individual person
# 
# Urban : Whether that person belongs to urban area or not
#  
# 

# In[1]:


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


# In[3]:


data=pd.read_csv("Fraud_check.csv")


# In[5]:


data.head()


# In[6]:


data[data.duplicated()]


# In[7]:


y=data['Taxable.Income']
Y=pd.DataFrame(y)
Y.describe()


# In[9]:


plt.figure(figsize=(20,5))
sns.boxplot(Y['Taxable.Income'],color= "plum")


# In[10]:


cat=[]
for i in data['Taxable.Income']:
    if i<= 30000:
        cat.append('Risky')

    else: 
        cat.append('No Risk')


# In[12]:


cat


# In[13]:


Y['cat']=pd.DataFrame(cat)
Y


# In[14]:


data['Class']=Y['cat']
data


# In[23]:


dataf=data.drop(['Taxable.Income'],axis=1)


# In[16]:


data['Class'].value_counts()


# In[17]:


label_encoder = preprocessing.LabelEncoder()
data['Undergrad']= label_encoder.fit_transform(data['Undergrad']) 
data['Marital.Status']= label_encoder.fit_transform(data['Marital.Status']) 
data['Work.Experience']= label_encoder.fit_transform(data['Work.Experience']) 
data['Urban']= label_encoder.fit_transform(data['Urban']) 
data['Class']= label_encoder.fit_transform(data['Class']) 


# In[21]:


sns.pairplot(data)


# In[25]:


dataf.head()


# In[26]:


x=data.iloc[:,0:5]
tar=data.iloc[:,5]


# In[27]:


tar.head()


# In[121]:


# Splitting data into training and testing data set
x_train, x_test,tar_train,tar_test = train_test_split(x,tar, test_size=0.1,random_state=18)


# In[122]:


x_train


# In[123]:


tar_train.value_counts()


# In[124]:


tar_test.value_counts()


# In[125]:


model=DecisionTreeClassifier(criterion='gini',min_samples_split=4,max_depth=5)
model.fit(x_train,tar_train)


# In[126]:


fn=['CompPrice','Income','Advertising','Population', 'Price', 'ShelveLoc','Age', 'Education','Urban','US']
cn=['high', 'low', 'moderate']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,3), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[136]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[128]:


from sklearn import metrics
metrics.accuracy_score(preds,tar_test)


# In[129]:


preds 


# In[137]:


pd.crosstab(tar_test,preds)


# In[139]:


np.mean(preds==tar_test)


# In[ ]:




