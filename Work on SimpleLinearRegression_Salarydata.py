#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression Model
# 
# **Description:**
# Building a simple linear regression model by performing EDA and to perform necessary transformations and select the best model parameters using Python.
# 
# **Data:**
# Salary data
# Features: Experience in years
# Target Variable: Salary earned
# 
# It is a supervised Machine Learning Model we need to perform Simple Linear Regression.

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#reading the dataset
Sal = pd.read_csv("Salary_Data.csv")
Sal.head()


# ### Preprocessing the data

# In[5]:


Sal.info() #null values
#Sal.describe()


# In[14]:


#ourtliers detection
plt.boxplot(Sal['Salary']) 


# In[15]:


plt.boxplot(Sal['YearsExperience']) 


# In[21]:


plt.scatter(Sal['YearsExperience'],Sal['Salary'],color='purple')


# In[22]:


plt.hist(Sal['Salary'],color='Orange')


# In[24]:


Sal['Salary'].corr(Sal['YearsExperience'])


# ### Building Model
# 
# In the given problem,
# y(target variable) = Salary
# 
# x(independent variable) = YearsExperience

# In[26]:


import statsmodels.formula.api as smf
Sal_model=smf.ols("Salary~YearsExperience",data=Sal).fit()
Sal_model.params


# In[27]:


Sal_model.summary()


# In[28]:


Sal_model.resid


# In[29]:


rmse=np.sqrt(np.mean(Sal_model.resid**2))
rmse


# In[30]:


Sal_model.rsquared


# In[31]:


pred=Sal_model.predict(Sal.iloc[:,0])
pred


# In[32]:


#best fit line for the model
import matplotlib.pylab as plt
plt.plot(Sal.iloc[:,0],pred,color='red')
plt.xlabel('Years of experience')
plt.ylabel('Predicted Salary')


# In[33]:


#best fit line for the model
import matplotlib.pylab as plt
plt.plot(Sal.iloc[:,0],pred,color='red')
plt.scatter(Sal['YearsExperience'],Sal['Salary'],color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Predicted Salary')


# In[34]:


pd.Series['Sal_model.rsquared','rmse']:[Sal_model.rsquared,rmse]


# In[18]:


results = {"model":pd.Series(["rmse","rsquared"]),
        "values":pd.Series([rmse,Sal_model.rsquared])}
         
result_final=pd.DataFrame(results)
result_final


# ### Conclusions:
# 
# Since rsquared is incredibly high we tend to consider the model developed to be the best model for the given dataset.
# 
# **Model Parameters**
# 
# Intercept          25792.200199
# 
# YearsExperience     9449.962321
# 
# Equation for finding salary = (25792.200199)+(9449.962321*YearsExperience)
# 
# Note: Since the adjusted r_Squared values are pretty satisfactory, we haven't done model improvement techniques

# In[ ]:




