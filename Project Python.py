
# coding: utf-8

# In[1]:


import os
import pandas as pd
import seaborn as sns
from random import randrange, uniform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics  


# In[2]:


os.chdir("C:/Users/BITTU/Desktop/Project 1")
os.getcwd()


# In[4]:


df_csv.columns


# In[6]:


#these are columns for correlation analysis
cnames_corr=['dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered','cnt']
#below features contains outliars
cnames_ol=['temp','atemp','hum','windspeed']


# In[7]:


#dropping outlier rows
for i in cnames_ol:
    q75,q25=np.percentile(df_csv.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25-(iqr*1.5)
    max=q75+(iqr*1.5)
    df_csv=df_csv.drop(df_csv[df_csv.loc[:,i]<min].index)
    df_csv=df_csv.drop(df_csv[df_csv.loc[:,i]>max].index)


# In[8]:


#saving into new dataset called df_csv_1
df_csv_1=df_csv.loc[:,cnames_corr]
df_csv_1.shape


# In[10]:


#new dataset called df_csv_2 for correlation analysis and also applied on onlynecessary features
df_csv_2=df_csv_1.iloc[:,0:12]


# In[ ]:


#below is correlation matrix
f, ax = plt.subplots(figsize=(7, 5))
#Generate correlation matrix
corr = df_csv_2.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[11]:


#below is the step to remove columns that contain hign +ve and -ve correlations
df_csv_1=df_csv_1.drop(['windspeed','hum','dteday','holiday','windspeed','atemp','casual','registered'],axis=1)
x=df_csv_1.drop('cnt',axis=1)
y=df_csv_1['cnt']


# In[12]:


#is to check column names of removing target feature
df_csv_1.columns


# In[13]:


#below is to divide dataset into train and test data sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[14]:


#Applying DT regression
fit_Dt=DecisionTreeRegressor(max_depth=50).fit(x_train,y_train)


# In[15]:


#storing predicted values into y_pred using above model called fit_Dt
y_pred=fit_Dt.predict(x_test)


# In[16]:


#to see Actual values and predicted values
df=pd.DataFrame({'Actual':y_test,'Prediction':y_pred})
df


# In[17]:


#caluculation of MAE MQE,RMSE
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


# In[18]:


#calculation of MAPE
def MAPE(x,y):
    mape=np.mean(np.abs((x-y)/x))*100
    return mape
#Percentage of error in out DT Regression Model
count=MAPE(y_test,y_pred)
count


# In[19]:


#Linear Regression from scratch
import os
import pandas as pd
import seaborn as sns
from random import randrange, uniform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import statsmodels.api as sm


# In[604]:


os.chdir("C:/Users/Rajashekar/Videos/python_project")
os.getcwd()


# In[605]:


df_csv=pd.read_csv("day.csv",sep=",")


# In[606]:


#these are columns for correlation analysis
cnames_corr=['dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered','cnt']
#below features contains outliars
cnames_ol=['temp','atemp','hum','windspeed']


# In[607]:


#dropping outlier rows
for i in cnames_ol:
    q75,q25=np.percentile(df_csv.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25-(iqr*1.5)
    max=q75+(iqr*1.5)
    df_csv=df_csv.drop(df_csv[df_csv.loc[:,i]<min].index)
    df_csv=df_csv.drop(df_csv[df_csv.loc[:,i]>max].index)


# In[608]:


#saving into new dataset called df_csv_1
df_csv_1=df_csv.loc[:,cnames_corr]
df_csv_1.shape


# In[609]:


#new dataset called df_csv_2 for correlation analysis and also applied on onlynecessary features
df_csv_2=df_csv_1.iloc[:,0:12]


# In[610]:


#below is correlation matrix
f, ax = plt.subplots(figsize=(7, 5))
#Generate correlation matrix
corr = df_csv_2.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[611]:


#below is the step to remove columns that contain hign +ve and -ve correlations
df_csv_1=df_csv_1.drop(['season','windspeed','dteday','holiday','hum','windspeed','atemp','casual','registered'],axis=1)
x=df_csv_1.drop('cnt',axis=1)
y=df_csv_1['cnt']


# In[612]:


#is to check column names of removing target feature
df_csv_1.columns


# In[613]:


#below is to divide dataset into train and test data sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[20]:


# Train the model using the training sets
model = sm.OLS(y_train, x_train).fit()


# In[21]:


#summary of above model
model.summary()


# In[22]:


#prediction of values of above mode
y_pred= model.predict(x_test) 


# In[23]:


#MAPE1 function
def MAPE1(x,y):
    mape=np.mean(np.abs((x-y)/x))*100
    return mape
#Error percentage
count=MAPE1(y_test,y_pred)
count

