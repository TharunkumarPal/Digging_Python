#!/usr/bin/env python
# coding: utf-8

# # MULTI LINEAR REGRESSION
# 
# **Description:** Building a Multi Variate linear regression model by performing EDA and to perform necessary transformations and select the best model parameters using Python.
# 
# **Data:** 50 startups data
# 
# Prepare a prediction model for profit of 50_startups data.
# 
# 
# **Features:** "R&D Spend"	"Administration"	"Marketing Spend"	"State"
# 
# **Target Variable:** Profit
# 
# It is a supervised Machine Learning Model we need to perform Multi Linear Regression

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# ## Preprocessing Step

# In[2]:


#Read the data 
startup = pd.read_csv("50_Startups.csv")
startup.head()


# In[14]:


#creating a new dataframe with the required features
startup.columns


# In[15]:


startup.info()


# In[7]:


data = startup.rename({'R&D Spend':'RDS','Administration':'Admin','Marketing Spend':'MS'},axis=1)
data.head()


# In[6]:


data[data.duplicated()]


# **^No duplicate entries are observed**

# In[8]:


data.describe()


# In[9]:


data.corr()


# In[26]:


sns.set_style(style='darkgrid')
sns.pairplot(data)


# In[27]:


model=smf.ols('Profit~RDS+Admin+MS',data=data).fit()


# In[29]:


model.params


# In[30]:


model.summary()


# **Inferences:**
# 
# Although rsquared and adj_rsquared are desirably as good, still we can observe the features 'Admin' and 'MS' have been insignificant, let's proceed to check the significance of those variables.

# In[32]:


#SLR for profit vs admin
profit_adm=smf.ols('Profit~Admin',data=data).fit()
profit_adm.summary()


# In[33]:


#SLR for price vs Doors
profit_MS=smf.ols('Profit~MS',data=data).fit()
profit_MS.summary()


# In[37]:


#mlr for profit vs MS+Admin
m1=smf.ols('Profit~MS+Admin',data=data).fit()
m1.summary()


# ## Multi Collinearity Problem
# 
# Since it can be observed that even after 'Admin','MS' are found to be insignificant still due to multicollinearity between the variables we can't exclude one of them. So VIF technique is used to identify which one to drop to improve the model.

# In[39]:


rsq_pro = smf.ols('Profit~RDS+Admin+MS',data=data).fit().rsquared  
vif_pro = 1/(1-rsq_pro) 

rsq_rd = smf.ols('RDS~Profit+Admin+MS',data=data).fit().rsquared  
vif_rd = 1/(1-rsq_rd) 

rsq_adm = smf.ols('Admin~Profit+RDS+MS',data=data).fit().rsquared  
vif_adm = 1/(1-rsq_adm)

rsq_ms = smf.ols('MS~Admin+Profit+RDS',data=data).fit().rsquared  
vif_ms = 1/(1-rsq_ms)

# Storing vif values in a data frame
d1 = {'Variables':['Proft','RDS','Admin','MS'],'VIF':[vif_pro,vif_rd,vif_adm,vif_ms]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[43]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.xlabel('Residual Quantiles (Theoretical)')
plt.ylabel('Actual Residuals')
plt.show()


# In[44]:


plt.scatter((model.fittedvalues),
           (model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[46]:


fig = plt.figure(figsize=(15,12))
fig = sm.graphics.plot_regress_exog(model, "RDS", fig=fig)
plt.show()


# In[47]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Admin", fig=fig)
plt.show()


# In[ ]:





# In[48]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "MS", fig=fig)
plt.show()


# ## Outliers Presence
# 
# Here we used influence plots using cook's distance method to identify the outliers.

# In[56]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[57]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[58]:


(np.argmax(c),np.max(c))


# In[59]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[60]:


k = data.shape[1]
n = data.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# It has appeared that all the data points lie within the leverage distance, but to understand if any accuracy can be improved by removing the data with highest cook's distance has been tried.

# In[62]:


data1=data.drop(data.index[[49]],axis=0).reset_index()
data1


# In[64]:


model_final=smf.ols('Profit~RDS+Admin+MS',data1).fit()


# In[65]:


model_final.summary()


# Since accuracy improved by the outlier treatment, we consider the model_final as our required model.

# ## New Prediction

# In[66]:


model_final.predict({'RDS':100000,'Admin':5000,'MS':3000})

