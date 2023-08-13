#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


#import dataset from sklearn library
from sklearn.datasets import load_boston


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


load_boston


# In[5]:


load_boston()


# In[6]:


df = load_boston()


# In[11]:


dataset = pd.DataFrame(df.data)


# In[12]:


dataset


# In[13]:


dataset.columns=df.feature_names


# In[14]:


dataset.head()


# In[16]:


#defining the dependent and independent features
X = dataset
y = df.target


# In[17]:


y


# In[18]:


#train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[19]:


#check the train dataset
X_train


# In[21]:


#standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[22]:


X_train = scaler.fit_transform(X_train)


# In[23]:


X_test  = scaler.transform(X_test)


# In[35]:


from sklearn.linear_model import LinearRegression
#cross-validation- It is a technique for evaluating machine learning models by training several models on subsets of the available input data and evaluating them on the complementary subset of the data.
from sklearn.model_selection import cross_val_score


# In[25]:


regression = LinearRegression()


# In[28]:


mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=5)


# In[29]:


np.mean(mse)


# In[30]:


#fit model to data
regression.fit(X,y)


# In[31]:


#prediction
reg_pred = regression.predict(X_test)


# In[32]:


reg_pred


# In[37]:


#visualizing the mse through kde plot
import seaborn as sns
sns.displot(reg_pred-y_test,kind='kde')


# In[36]:


from sklearn.metrics import r2_score


# In[38]:


score=r2_score(reg_pred,y_test)


# In[39]:


score

