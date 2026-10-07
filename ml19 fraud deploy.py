#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Loading Datset
df=pd.read_csv('fraud_data.csv', sep=';') #by default 'space' is the separator but here data is separated by colon;
df


# In[3]:


df.duplicated().sum()


# In[4]:


df.isnull().sum()


# In[5]:


df.size


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.boxplot()


# In[9]:


#outlier capping 
for i in df.select_dtypes(include=['int64', 'float64']).columns:
    q1=df[i].quantile(0.25)
    q3=df[i].quantile(0.75)
    IQR=q3-q1
    LB=q1-1.5*IQR
    UB=q3+1.5*IQR
    df[i]=df[i].clip(LB,UB)


# In[10]:


df.boxplot()
plt.show()


# In[11]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for i in df.select_dtypes(include=['string']).columns:
    df[i]=le.fit_transform(df[i])


# In[12]:


df


# In[13]:


df.rename(columns={'6m_avg_chbk_amt': 'avg_chbk_amt','6m_chbk_freq': 'chbk_freq'},inplace=True)


# In[14]:


feat=df.drop(columns='is_fradulent')
tar=df['is_fradulent']


# In[15]:


#feature selection... f_classifier
from sklearn.feature_selection import f_classif

fclass=f_classif(feat,tar) # .index[0:5] to get top 5 values
f_scores=pd.Series(fclass[0],index= feat.columns).sort_values() # 0 will give f scores
print(f_scores)


# In[16]:


f_scores.plot(kind='barh')
plt.show()


# In[17]:


df1=df[['avg_amount_days', 'is_declined', 'foreign_transaction', 'transaction_amount','high_risk_countries','is_fradulent']]


# In[18]:


x=df1.drop(columns='is_fradulent')
y=df1['is_fradulent']


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[20]:


x_train.info()


# In[21]:


#applying standard scaler
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
cols=['avg_amount_days','transaction_amount']
x_train[cols]=ss.fit_transform(x_train[cols]) #for training data, fit_transform will be used
x_test[cols]=ss.transform(x_test[cols]) #for testing data, only transform will be used


# In[22]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(x_train,y_train)

pred_train=lr.predict(x_train)
pred_test=lr.predict(x_test)

from sklearn.metrics import *
print("Train Acc is ",accuracy_score(y_train,pred_train))
print(classification_report(y_train,pred_train))
print("Test Acc is ",accuracy_score(y_test,pred_test))
print(classification_report(y_test,pred_test))


# In[23]:


import joblib
joblib.dump(lr, "logmodel.pkl")


# In[24]:


# Save scaler
joblib.dump(ss, "stscaler.pkl")


# In[25]:


x_train


# In[26]:


x_train.columns


# In[27]:


x_train


# In[ ]:




