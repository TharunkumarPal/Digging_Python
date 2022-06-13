#!/usr/bin/env python
# coding: utf-8

# ## Work on  Emotion Mining

# In[1]:


get_ipython().system('pip install future')


# In[2]:


import codecs
import re
import copy
import collections
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
import matplotlib
    
get_ipython().run_line_magic('matplotlib', 'inline')


from __future__ import division
import os
from nltk.corpus import twitter_samples


# In[3]:


nltk.download('stopwords')


# In[4]:


from nltk.corpus import stopwords


# In[5]:


with codecs.open("positive-words.txt", "r", encoding="utf-8") as p:
    pos = p.read()
    print(pos)


# In[6]:


nltk.download('twitter_samples')


# In[7]:


from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')


# In[8]:


import nltk
nltk.download('punkt')


# In[9]:


from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')


# In[10]:


get_ipython().system('pip3 install beautifulsoup4')


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests


# In[12]:


df = pd.read_csv("amazon_products.csv")
df


# In[13]:


df.shape


# In[14]:


df.head(61)


# In[15]:


df['Rating'] = df['Rating'].apply(lambda x:x)
df['Rating'] = pd.to_numeric(df['Rating'])


# In[16]:


df["Price"] = df["Price"].replace('₹', '')
df["Price"] = df["Price"].replace(',', '')


# In[17]:


df['Price'] = df['Price'].apply(lambda x: x)


# In[18]:


df['Price'] = df['Price'].astype(int)


# In[19]:


df["Num_Customers_Rated"] = df["Num_Customers_Rated"].replace(',', '')


# In[20]:


df['Num_Customers_Rated'] = pd.to_numeric(df['Num_Customers_Rated'], errors='ignore')


# In[21]:


df.head()


# In[22]:


df.dtypes


# In[23]:


df.replace(str(0), np.nan, inplace=True)
df.replace(0, np.nan, inplace=True)


# In[24]:


count_nan = len(df) - df.count()


# In[25]:


count_nan


# In[26]:


df = df.dropna()


# In[27]:


data = df.sort_values(["Price"], axis=0, ascending=False)[:15]


# In[28]:


data


# In[29]:


pip install typing-extensions --upgrade


# In[30]:


pip install bokeh


# In[31]:


from bokeh.models import ColumnDataSource
from bokeh.transform import dodge
import math
from bokeh.io import curdoc
curdoc().clear()
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import Legend
output_notebook()


# In[32]:


p = figure(x_range=data.iloc[:,2], plot_width=80, plot_height=60, title="Top Rated Books with more than 1000 Customers Rating", toolbar_location=None, tools="")

p.vbar(x=data.iloc[:,0], top=data.iloc[:,5], width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0
p.xaxis.major_label_orientation = math.pi/2


# In[37]:


show(p)


# In[38]:


data = df.sort_values(["Num_Customers_Rated"], axis=0, ascending=False)[:20]
data


# In[39]:


from bokeh.transform import factor_cmap
from bokeh.models import Legend
from bokeh.palettes import Dark2_5 as palette
import itertools
from bokeh.palettes import d3
#colors has a list of colors which can be used in plots
colors = itertools.cycle(palette)

palette = d3['Category20'][20]


# In[40]:


index_cmap = factor_cmap('Author', palette=palette,
                         factors=data["Author"])
p = figure(plot_width=700, plot_height=700, title = "Top Authors: Rating vs. Customers Rated")
p.scatter('Rating','Num_Customers_Rated',source=data,fill_alpha=0.6, fill_color=index_cmap,size=20,legend='Author')
p.xaxis.axis_label = 'RATING'
p.yaxis.axis_label = 'CUSTOMERS RATED'
p.legend.location = 'top_left'


# In[41]:


show(p)


# In[ ]:




