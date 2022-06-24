#!/usr/bin/env python
# coding: utf-8

# # Neural Networks 
# 
# The dataset contains 36733 instances of 11 sensor measures aggregated over one hour (by means of average or sum) from a gas turbine. 
# 
# The Dataset includes gas turbine parameters (such as Turbine Inlet Temperature and Compressor Discharge pressure) in addition to the ambient variables.
# 
# 
# 
# **Problem Statement:** predicting turbine energy yield (TEY) using ambient variables as features.
# 
# **Attribute Information:**
# 
# The explanations of sensor measurements and their brief statistics are given below.
# 
# Variable (Abbr.) Unit Min Max Mean
# 
# Ambient temperature (AT) C â€“6.23 37.10 17.71
# 
# Ambient pressure (AP) mbar 985.85 1036.56 1013.07
# 
# Ambient humidity (AH) (%) 24.08 100.20 77.87
# 
# Air filter difference pressure (AFDP) mbar 2.09 7.61 3.93
# 
# Gas turbine exhaust pressure (GTEP) mbar 17.70 40.72 25.56
# 
# Turbine inlet temperature (TIT) C 1000.85 1100.89 1081.43
# 
# Turbine after temperature (TAT) C 511.04 550.61 546.16
# 
# Compressor discharge pressure (CDP) mbar 9.85 15.16 12.06
# 
# Turbine energy yield (TEY) MWH 100.02 179.50 133.51
# 
# Carbon monoxide (CO) mg/m3 0.00 44.10 2.37
# 
# Nitrogen oxides (NOx) mg/m3 25.90 119.91 65.29

# In[2]:


import tensorflow  as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras


# In[3]:


# fix random seed for reproducibility
seed = 7
data=pd.read_csv("gas_turbines.csv", delimiter=",")
data


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


# Check and drop the duplicate values
data[data.duplicated()]


# In[11]:


X = data.drop(axis=0,columns="TEY").values
Y = data["TEY"].values


# In[16]:


sns.pairplot(data)


# In[21]:


sns.distplot(Y)


# In[13]:


# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(22, input_dim=10, activation='relu'))
model.add(tf.keras.layers.Dense(10,  activation='relu'))
model.add(tf.keras.layers.Dense(7,  activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1))


# In[14]:


model.summary()


# In[15]:


# Compile model
model.compile(loss ='mse', optimizer='adam', metrics=['mse']) #adam is kind of Graddescent


# In[17]:


# Fit the model
history=model.fit(X, Y, validation_split=0.20, epochs=50, batch_size=100)


# In[18]:


model.save_weights("mywt.kmw")


# In[21]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))


# In[22]:


# Visualize training history

# list all data in history
model.history.history.keys()
import matplotlib.pyplot as plt


# In[23]:


history.history.keys()


# In[25]:


# summarize history for accuracy
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[111]:




