#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta as ta
from def_funktionen import * 
from azure.cosmos import CosmosClient

key = "xF2uR8yl9kvI1801q5rn68bJ6QBYb0Kz4ur99MdT90p26RA0XqhrCGTBgP8ivkJKmt3mk1nggZtRACDbsapTKA=="
url = "https://apptrading.documents.azure.com:443/"
database_name = "database"
container_name = "Container1"

client = CosmosClient(url, credential=key)
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)





# # Datensatz hochladen

# In[2]:


data=yf.download(tickers="EURUSD=X", start='2000-01-01',end='2023-11-09')
data.tail(10)


# # Adding Indicators

# In[3]:


data["SuperTrend"]=ta.supertrend(data.High,data.Low,data.Close,length=7,multiplier=3,offset=0)["SUPERTd_7_3.0"]


data["RSI"]=ta.rsi(data.Close,length=15)
data["EMAF"]=ta.ema(data.Close,length=20);
data["EMAM"]=ta.ema(data.Close,length=100);
data["EMAS"]=ta.ema(data.Close,length=150);


# In[4]:


data["CorrF"], data["CorrS"]=BCL2ECTS(data.High,data.Low,data.Open,data.Close,fast_length=20,slow_length=40,buy_threshold=0.5,sell_threshold=0)
data["TII"]=TII(data.High,data.Low,data.Open,data.Close,majorLength=60,minorLength=30,upperLevel=80,lowerLevel=20)
halftrend_out,halftrend_df=halftrend(data.High,data.Low,data.Open,data.Close)
data["HalfTrend"]=halftrend_out.direction
data["ASO"]=aso(data.High,data.Low,data.Open,data.Close,length=10)
data["TPR_AM+"],data["TPR_AM-"],data["TPR_FM"]=TPR(data.High,data.Low,data.Open,data.Close,length=14)


# In[5]:


data['Target'] = data['Adj Close']-data.Open
data['Target'] = data.SuperTrend.shift(-1)

data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace = True)
data.drop(['Volume', 'Close', 'Date','SuperTrend'], axis=1, inplace=True)
data.tail(10)


# In[6]:


data.columns


# In[7]:


data_set=data.iloc[:,0:20]
pd.set_option('display.max_columns',None)



data_set


# In[8]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)
print(data_set_scaled)


# In[9]:


# multiple feature from data provided to the model
X = []
#print(data_set_scaled[0].size)
#data_set_scaled=data_set.values
backcandles = 50
for j in range(17):#data_set_scaled[0].size):#2 columns are target not X
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
        X[j].append(data_set_scaled[i-backcandles:i, j])


# In[10]:


X=np.moveaxis(X, [0], [2])


#Erase first elements of y because of backcandles to match X length
#del(yi[0:backcandles])
#X, yi = np.array(X), np.array(yi)
# Choose -1 for last column, classification else -2...
X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-3])
y=np.reshape(yi,(len(yi),1))
#y=sc.fit_transform(yi)
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# In[11]:


# split data into train test sets
splitlimit = int(len(X)*0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]


# In[12]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
#tf.random.set_seed(20)
np.random.seed(10)


# In[13]:


lstm_input = Input(shape=(backcandles, 17), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)

inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)


# In[14]:


y_pred = model.predict(X_test)
#y_pred=np.where(y_pred > 0.43, 1,0)
#for i in range(len(y_pred)):
    #print(y_pred[i], y_test[i])
    
    
# Your JSON document to be inserted
json_document = {
    "id": "10",  # Farklı bir id kullanın
    "category": "personal",
    "name": "Long",  # Değerinizi buraya ekleyin
    "description": "your_description_here",  # İsteğe bağlı olarak açıklama ekleyin
    "isComplete": False
}

# Insert JSON document into Cosmos DB
container.upsert_item(body=json_document)

# In[15]:


#plt.figure(figsize=(16,8))
#plt.plot(y_test, color = 'black', label = 'Test')
#plt.plot(y_pred, color = 'green', label = 'pred')
#plt.legend()
#plt.show()




#y_prediction=model.predict(versuch1)


# 

# In[ ]:







