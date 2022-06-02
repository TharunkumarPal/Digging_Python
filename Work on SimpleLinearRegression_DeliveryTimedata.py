#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression Model
# 
# **Description:** Building a simple linear regression model by performing EDA and to perform necessary transformations and select the best model parameters using Python.
# 
# **Data:** 
# Delivery time data 
# We need to predict the Delivery time using the given sorting time
# 
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
delivery = pd.read_csv("delivery_time.csv")
delivery.columns


# ### Preprocessing 

# In[3]:


delivery.info()


# In[4]:


delivery.rename(columns={'Delivery Time': 'Del_time','Sorting Time': 'Sort_time'}, inplace=True)
delivery['Del_time']


# In[5]:


delivery.describe()


# In[6]:


plt.boxplot(delivery.Sort_time)


# In[7]:


plt.boxplot(delivery.Del_time)


# In[8]:


plt.hist(delivery.Sort_time,color='Orange')


# In[9]:


plt.plot(delivery.Del_time,delivery.Sort_time,"bo", color='purple')
plt.xlabel("Delivery Time")
plt.ylabel("Sorting Time")


# In[10]:


delivery.Del_time.corr(delivery.Sort_time)


# ### Model Building
# **Building model #1**
# 
# We completed the preprocessing, So let's start model building
# 
# In this step, let's create a linear regression model y vs x 

# In[11]:


import statsmodels.formula.api as smf
del_model1=smf.ols("Del_time~Sort_time",data=delivery).fit()
del_model1.params


# In[12]:


del_model1.summary()


# In[13]:


del_model1.resid 


# In[14]:


del_model1.resid_pearson


# In[15]:


pred1 = del_model1.predict(delivery.iloc[:,1])
pred1
#pd.set_option("display.max_rows", 100) ----> restricts o/p to only 100 rows
pred1


# In[16]:


rmse_lin = np.sqrt(np.mean((np.array(delivery['Del_time'])-np.array(pred1))**2))
rmse_lin


# In[17]:


import matplotlib.pylab as plt
plt.scatter(x=delivery['Sort_time'],y=delivery['Del_time'],color='green')


# In[18]:


#best fit line for the model #1
plt.plot(delivery['Sort_time'],pred1,color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Predicted Del_Time')


# In[19]:


plt.scatter(x=delivery['Sort_time'],y=delivery['Del_time'],color='green')
plt.plot(delivery['Sort_time'],pred1,color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Predicted Del_Time')


# In[20]:


del_model1.rsquared, rmse_lin


# **Building model #2 (Log model)**
# 
# Since the r-squared value obtained in previous step is not good enough, 
# 
# Let's go for alternate regression technique like y vs log(x)

# In[21]:


del_model2 = smf.ols('Del_time~np.log(Sort_time)',data=delivery).fit()
del_model2.params


# In[22]:


del_model2.summary()


# In[23]:


pred2 = del_model2.predict(delivery.iloc[:,1])
pred2
pd.set_option("display.max_rows", 109) 
pred2
rmse_log = np.sqrt(np.mean((np.array(delivery['Del_time'])-np.array(pred2))**2))
rmse_log


# In[24]:


plt.scatter(x=delivery['Sort_time'],y=delivery['Del_time'],color='green')
plt.plot(delivery['Sort_time'],pred2,color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Predicted Del_Time for model 2')


# In[25]:


del_model2.rsquared, rmse_log


# **Building model #3 (Exponential Model)**
# 
# Since the r-squared value obtained in previous step is also not good enough,
# 
# Let's go for alternate regression technique like log(y) vs x or y vs exp(x)

# In[26]:


del_model3=smf.ols('np.log(Del_time)~Sort_time',data=delivery).fit()
del_model3.params


# In[27]:


del_model3.summary()


# In[28]:


pred3 = del_model3.predict(delivery.iloc[:,1])
pred3 #here y vales obtained are log(y) hence to find actual expected y values we need to take exponent of pred3
pred3_exp=np.exp(pred3)
pred3_exp


# In[29]:


pd.set_option("display.max_rows", 109) 
pred3_exp
rmse_exp = np.sqrt(np.mean((np.array(delivery['Del_time'])-np.array(pred3_exp))**2))
rmse_exp


# In[30]:


plt.scatter(x=delivery['Sort_time'],y=delivery['Del_time'],color='green')
plt.plot(delivery['Sort_time'],pred3_exp,color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Predicted Del_Time for exp model')


# **Building model #4**
# 
# Since the r-squared value obtained in previous step is not good enough, 
# 
# Let's go for alternate regression technique like Log(Y)= X+Sq(X)

# In[31]:


delivery["Sort_timesq"] = delivery.Sort_time*delivery.Sort_time
delivery


# In[32]:


del_model4=smf.ols('np.log(Del_time)~Sort_time+Sort_timesq',data=delivery).fit()
del_model4.params


# In[33]:


del_model4.summary()


# In[34]:


pred_quad = del_model4.predict(delivery)
pred4=np.exp(pred_quad) 
pred4
rmse_quad = np.sqrt(np.mean((np.array(delivery['Del_time'])-np.array(pred4))**2))
rmse_quad 


# In[35]:


results = {"model":pd.Series(["rmse_lin","rmse_exp","rmse_log","rmse_quad"]),
        "rmse_values":pd.Series([rmse_lin,rmse_exp,rmse_log,rmse_quad]),
         "rsquared_values":pd.Series([del_model1.rsquared,del_model2.rsquared,del_model3.rsquared,del_model4.rsquared])}


# In[36]:


result_final=pd.DataFrame(results)
result_final


# ### Conclusions:
# 
# It is found that Model_quad has given us better r_squared values relative to other models so model_quad stands out to be our final model.
# 
# **Model Parameters**
# 
# Intercept= 1.699704
# 
# Sort_time= 0.265922
# 
# Sort_timesq= -0.012841 
# 
# Equation for finding 
# **log(delivery time) = (1.699704)+(0.26*YearsExperience)-0.012*pow((Sort_time),2)**
# 
# Note: Since we couldn't reach minimum accuracy of 85%, let's settle with the maximum accuracy obtained among the 4 models.

# In[ ]:




