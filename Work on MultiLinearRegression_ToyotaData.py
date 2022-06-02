#!/usr/bin/env python
# coding: utf-8

# # MULTI LINEAR REGRESSION
# 
# **Description:** Building a Multi Variate linear regression model by performing EDA and to perform necessary transformations and select the best model parameters using Python.
# 
# **Data:** Toyota Corolla data
# 
# Consider only the below columns and prepare a prediction model for predicting Price.
# 
# Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
# 
# **Features:** "Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"
# 
# **Target Variable:** Price
# 
# It is a supervised Machine Learning Model we need to perform Multi Linear Regression

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[2]:


#Read the data
corolla = pd.read_csv("ToyotaCorolla.csv")
corolla.head(3)


# In[3]:


#creating a new dataframe with the required features
corolla.columns


# In[4]:


corolla.info()


# In[5]:


corolla2=pd.concat([corolla.iloc[:,2:4],corolla.iloc[:,6:7],corolla.iloc[:,8:9],corolla.iloc[:,12:14],corolla.iloc[:,15:18]],axis=1)
corolla2


# In[6]:


corolla3=corolla2.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
corolla3


# In[7]:


corolla3[corolla3.duplicated()]


# In[8]:


c_final=corolla3.drop_duplicates().reset_index(drop=True)
c_final


# In[9]:


c_final.describe()


# In[10]:


c_final.corr()


# In[11]:


sns.set_style(style='darkgrid')
sns.pairplot(c_final)


# In[12]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=c_final).fit()


# In[13]:


model.params


# In[14]:


model.summary()


# **Inferences:**
# 
# Although rsquared and adj_rsquared are desirably as good, still we can observe the features 'cc' and 'Doors' have been insignificant, let's check the significance of those variables.

# ### Understanding the significance of features whose p<0.05 

# In[15]:


#SLR for price vs CC
price_cc=smf.ols('Price~CC',data=c_final).fit()
price_cc.summary()


# In[16]:


#SLR for price vs Doors
price_d=smf.ols('Price~Doors',data=c_final).fit()
price_d.summary()


# In[17]:


price_cd=smf.ols('Price~CC+Doors',data=c_final).fit()
price_cd.summary()


# In[18]:


rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=c_final).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=c_final).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=c_final).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=c_final).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=c_final).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=c_final).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=c_final).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=c_final).fit().rsquared
vif_WT=1/(1-rsq_WT)

# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# ## Residual Analysis
# 
# **Q-Q Plots**
# 
# To understand the normality of the residuals obtained in each model using Q-Q plots.

# In[19]:


sm.qqplot(model.resid,line='q')  
plt.title("Normal Q-Q plot of residuals")
plt.xlabel("Standard Values")
plt.xlabel("Residuals")
plt.show()


# In[20]:


sorted(model.resid)


# In[21]:


list(np.where(model.resid>6000))


# In[22]:


list(np.where(model.resid<-6000))


# In[23]:


def standard_values(vals) : return (vals-vals.mean())/vals.std()


# In[24]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()


# ### Residual Vs Regressors
# 

# In[25]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)
plt.show()


# In[26]:


#Age+KM+HP+CC+Doors+Gears+QT+Weight
fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "KM", fig=fig)
plt.show()


# In[27]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "HP", fig=fig)
plt.show()


# In[28]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "CC", fig=fig)
plt.show()


# In[29]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "Doors", fig=fig)
plt.show()


# In[30]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "Gears", fig=fig)
plt.show()


# In[31]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "QT", fig=fig)
plt.show()


# In[32]:


fig = plt.figure(figsize=(15,10))
fig = sm.graphics.plot_regress_exog(model, "Weight", fig=fig)
plt.show()


# ## Model Deletion Diagnostics
# **Detecting Influencers/Outliers
#  Cook’s Distance**

# In[33]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[34]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 15))
plt.stem(np.arange(len(c_final)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[35]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# ### High Influence points

# In[36]:


from statsmodels.graphics.regressionplots import influence_plot
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[37]:


k = c_final.shape[1]
n = c_final.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# **Observation:**
# 
# From the observed leverage value
# 
# It is understood that the points with index 990,220,959,80 are exceeding the leverage cutoff.
# 
# Hence they are considered to be outliers and need to be treated for improving the model accuracy.

# In[38]:


extreme_val= c_final[c_final.index.isin([80,990,220,959])]
a=extreme_val.drop(['Price'],axis=1)
a


# **Note:**
# Since it is seen that multiple extreme values are observed we try to figure out the best model by excluding each of the extreme values and building the model gain.

# In[39]:


while model.rsquared < 0.93:
    for c in [np.max(c)>0.5]:
        model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=c_final).fit()
        (c,_)=model.get_influence().cooks_distance
        c
        np.argmax(c) 
        np.max(c)
        c_final=c_final.drop(c_final.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        c_final
    else:
        final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=c_final).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)


# In[40]:


final_model.summary()


# Hence the best model with 93%  accuracy has been developed.

# ## New Prediction

# In[41]:


new_data=pd.DataFrame({'Age':36,"KM":14567,"HP":100,"CC":1300,"Doors":4,"Gears":5,"QT":78,"Weight":1111},index=[0])
new_data
final_model.predict(new_data)


# In[42]:


final_model.predict(a)


# In[ ]:




