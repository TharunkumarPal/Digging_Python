#!/usr/bin/env python
# coding: utf-8

# # Neural Networks 
# 
# **Problem Statement:** PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS

# In[63]:


import tensorflow  as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras


# In[37]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
data=pd.read_csv("forestfires.csv", delimiter=",")


# In[38]:


data.shape


# In[39]:


data.info()


# In[40]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
label_encoder = preprocessing.LabelEncoder()
data['month'] = label_encoder.fit_transform(data['month'])
data['day'] = label_encoder.fit_transform(data['day'])
data['size_category'] = label_encoder.fit_transform(data['size_category'])
data


# In[68]:


# Check and drop the duplicate values
data[data.duplicated()]


# In[72]:


#drop the duplicate value
data_tem= data.drop_duplicates()


# In[73]:


data_tem


# In[75]:


data_tem.info()


# In[79]:


X = data_tem.iloc[:,0:11]
Y = data_tem.iloc[:,30]


# In[80]:


sns.pairplot(X)


# In[81]:


sns.countplot(Y)


# In[100]:


# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(22, input_dim=11, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10,  activation='relu'))
model.add(tf.keras.layers.Dense(7,  activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[101]:


model.summary()


# In[102]:


# Compile model
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #adam is kind of Graddescent


# In[103]:


# Fit the model
history=model.fit(X, Y, validation_split=0.20, epochs=50, batch_size=100)


# In[104]:


model.save_weights("mywt.kmw")


# In[105]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[106]:


# Visualize training history

# list all data in history
model.history.history.keys()
import matplotlib.pyplot as plt


# In[107]:


history.history.keys()


# In[108]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
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




