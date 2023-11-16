#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta as ta
from def_funktionen import * 


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
print(X)
print(X.shape)
print(y)
print(y.shape)


# In[11]:


# split data into train test sets
splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train)


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
for i in range(len(y_pred)):
    print(y_pred[i], y_test[i])
    
    


# In[15]:


plt.figure(figsize=(16,8))
plt.plot(y_test, color = 'black', label = 'Test')
plt.plot(y_pred, color = 'green', label = 'pred')
plt.legend()
plt.show()


# versuch1=[[[0.19935617, 0.20059464, 0.66867549, 0.19916487, 0.        ,
#          0.38968501, 0.18339111, 0.15095534, 0.11869483, 0.05082713,
#          0.15596944, 0.        , 0.        , 0.30573047, 0.02028806,
#          0.97971194, 0.42440619, 0.44858363],
#         [0.20927464, 0.19969075, 0.66742476, 0.20907382, 0.        ,
#          0.49598236, 0.18393445, 0.15108281, 0.11891966, 0.09630163,
#          0.11700249, 0.        , 0.        , 0.33277605, 0.02170917,
#          0.97829083, 0.42571681, 0.44858363],
#         [0.19542444, 0.19042385, 0.6640366 , 0.19523691, 0.        ,
#          0.3740292 , 0.1830147 , 0.15088696, 0.11891238, 0.0927289 ,
#          0.07630486, 0.        , 0.        , 0.34895964, 0.02500009,
#          0.97499991, 0.42633829, 0.44858363],
#         [0.18473777, 0.17914011, 0.66325397, 0.1845605 , 0.        ,
#          0.29996245, 0.18109357, 0.15044747, 0.1187284 , 0.09172649,
#          0.05123837, 0.        , 0.        , 0.34580971, 0.02881206,
#          0.97118794, 0.42695977, 0.44858363],
#         [0.18765208, 0.17757963, 0.65886641, 0.18747201, 0.        ,
#          0.33026896, 0.17965237, 0.15008419, 0.11859507, 0.09681201,
#          0.03497089, 0.        , 0.        , 0.34771408, 0.03262404,
#          0.96737596, 0.43366075, 0.44858363],
#         [0.17695439, 0.16956553, 0.65870749, 0.17678459, 0.        ,
#          0.26286181, 0.17725833, 0.14948033, 0.11828654, 0.09072638,
#          0.02006728, 0.        , 0.        , 0.34513537, 0.03643601,
#          0.96356399, 0.43228573, 0.44858363],
#         [0.17664805, 0.16736983, 0.65755168, 0.17647854, 0.        ,
#          0.26104347, 0.17506108, 0.14888133, 0.11797702, 0.08941732,
#          0.01826434, 0.        , 0.        , 0.35590292, 0.03895356,
#          0.96104644, 0.43132864, 0.44858363],
#         [0.17251601, 0.16860296, 0.65804221, 0.17235047, 0.        ,
#          0.23615373, 0.17265202, 0.14819849, 0.11760325, 0.08918455,
#          0.01821225, 0.        , 0.        , 0.36966624, 0.04185971,
#          0.95814029, 0.42588116, 0.44858363],
#         [0.1756941 , 0.17092934, 0.65912395, 0.17552551, 0.        ,
#          0.27362567, 0.17079626, 0.14760279, 0.117287  , 0.08854352,
#          0.01808125, 0.        , 0.        , 0.37715925, 0.04476585,
#          0.95523415, 0.42656247, 0.44858363],
#         [0.18074649, 0.17183979, 0.65877557, 0.18057305, 0.        ,
#          0.33112367, 0.16963207, 0.14713589, 0.11705853, 0.09233323,
#          0.01806868, 0.        , 0.        , 0.37742578, 0.0433213 ,
#          0.9566787 , 0.43296067, 0.44858363],
#         [0.18119919, 0.17227702, 0.65951806, 0.18102532, 0.        ,
#          0.33626793, 0.16862489, 0.14668873, 0.11684057, 0.10048695,
#          0.01739074, 0.        , 0.        , 0.35172827, 0.04187675,
#          0.95812325, 0.43172646, 0.44858363],
#         [0.17821656, 0.17063817, 0.65540119, 0.17804555, 0.        ,
#          0.31288482, 0.1674097 , 0.14618134, 0.11657615, 0.0976302 ,
#          0.01564245, 0.        , 0.        , 0.32106905, 0.03931579,
#          0.96068421, 0.43051395, 0.44858363],
#         [0.16301776, 0.16004421, 0.65405443, 0.16286133, 0.        ,
#          0.21250706, 0.16476148, 0.14533198, 0.1160638 , 0.08549088,
#          0.01609582, 0.        , 0.        , 0.29765033, 0.04158014,
#          0.95841986, 0.42608764, 0.44858363],
#         [0.16788701, 0.16169862, 0.65548368, 0.16772592, 0.        ,
#          0.26981707, 0.16286164, 0.14461221, 0.11563879, 0.07884903,
#          0.01570177, 0.        , 0.        , 0.29453862, 0.04384448,
#          0.95615552, 0.42804421, 0.44858363],
#         [0.17170991, 0.1650172 , 0.65680581, 0.17154514, 0.        ,
#          0.31308542, 0.1615323 , 0.14399524, 0.11528265, 0.0862291 ,
#          0.01598532, 0.        , 0.        , 0.29315229, 0.04610883,
#          0.95389117, 0.42913926, 0.44858363],
#         [0.17004638, 0.16622895, 0.65676818, 0.16988321, 0.        ,
#          0.30121203, 0.16016005, 0.14335195, 0.11490371, 0.08385573,
#          0.01615765, 0.        , 0.        , 0.30293106, 0.04861999,
#          0.95138001, 0.42021534, 0.44858363],
#         [0.16550214, 0.15783726, 0.65302543, 0.16534333, 0.        ,
#          0.26920448, 0.15845542, 0.14261616, 0.11445461, 0.0839759 ,
#          0.01608366, 0.        , 0.        , 0.30234667, 0.04434894,
#          0.95565106, 0.4197362 , 0.44858363],
#         [0.16683599, 0.15751477, 0.65277237, 0.1666759 , 0.        ,
#          0.28600736, 0.15704907, 0.14192583, 0.11403352, 0.07078826,
#          0.0139039 , 0.        , 0.        , 0.30270539, 0.03992331,
#          0.96007669, 0.4264521 , 0.44858363],
#         [0.16479144, 0.1549754 , 0.65026649, 0.16463331, 0.        ,
#          0.27068694, 0.15556831, 0.14120181, 0.11358419, 0.05864667,
#          0.01396707, 0.        , 0.        , 0.2979824 , 0.03549769,
#          0.96450231, 0.41752818, 0.44858363],
#         [0.15588964, 0.14766796, 0.64966128, 0.15574006, 0.        ,
#          0.20955268, 0.15332147, 0.14028596, 0.11299355, 0.05133361,
#          0.01344947, 0.        , 0.        , 0.28443055, 0.03314438,
#          0.96685562, 0.41704904, 0.44858363],
#         [0.15241846, 0.14212818, 0.64576   , 0.1522722 , 0.        ,
#          0.18811863, 0.15093491, 0.13930785, 0.1123533 , 0.05126079,
#          0.01354686, 0.        , 0.        , 0.28599576, 0.03229031,
#          0.96770969, 0.41273748, 0.44858363],
#         [0.14236819, 0.14290233, 0.64462434, 0.14223158, 0.        ,
#          0.13289621, 0.1477515 , 0.13811633, 0.11155527, 0.04444944,
#          0.01281174, 0.        , 0.        , 0.28319501, 0.03198242,
#          0.96801758, 0.41225833, 0.44858363],
#         [0.15131722, 0.14908539, 0.64902765, 0.15117202, 0.        ,
#          0.24850749, 0.14578319, 0.13715567, 0.11091586, 0.04544305,
#          0.01074574, 0.        , 0.        , 0.29625504, 0.03167453,
#          0.96832547, 0.4119139 , 0.44858363],
#         [0.15175405, 0.1448406 , 0.64468243, 0.15160844, 0.        ,
#          0.25386473, 0.14404686, 0.13622415, 0.11029214, 0.04495699,
#          0.01077212, 0.        , 0.        , 0.30478086, 0.03405136,
#          0.96594864, 0.40994201, 0.44858363],
#         [0.13852308, 0.1296325 , 0.64185171, 0.13839016, 0.        ,
#          0.17673479, 0.14112764, 0.13500464, 0.1094578 , 0.0462576 ,
#          0.01171448, 0.        , 0.        , 0.28699434, 0.03794084,
#          0.96205916, 0.4029336 , 0.44858363],
#         [0.1371995 , 0.13562995, 0.64210414, 0.13706785, 0.        ,
#          0.16986616, 0.13835157, 0.13377862, 0.10861262, 0.03651125,
#          0.01165305, 0.        , 0.        , 0.26863941, 0.04183032,
#          0.95816968, 0.40315188, 0.44858363],
#         [0.14264467, 0.13632816, 0.64522082, 0.14250779, 0.        ,
#          0.23880743, 0.13639475, 0.13270299, 0.10786871, 0.03051227,
#          0.01158786, 0.        , 0.        , 0.25386224, 0.04089448,
#          0.95910552, 0.41173137, 0.44858363],
#         [0.14883993, 0.14597042, 0.64410138, 0.14869711, 0.        ,
#          0.31124662, 0.1352556 , 0.13179215, 0.10723714, 0.0296639 ,
#          0.01270566, 0.        , 0.        , 0.24936688, 0.03995865,
#          0.96004135, 0.41138693, 0.44858363],
#         [0.15210371, 0.14255022, 0.64657774, 0.15195776, 0.        ,
#          0.34748283, 0.13455752, 0.13097494, 0.10666793, 0.04437764,
#          0.0150245 , 0.        , 0.        , 0.26441125, 0.03567685,
#          0.96432315, 0.4110425 , 0.44858363],
#         [0.15388934, 0.14933363, 0.64885842, 0.15374167, 0.        ,
#          0.3673733 , 0.13410788, 0.13021527, 0.1061358 , 0.07343504,
#          0.01801092, 0.01791361, 1.        , 0.29748822, 0.03139505,
#          0.96860495, 0.41962198, 0.44858363],
#         [0.15794782, 0.15153608, 0.65128747, 0.15779626, 0.        ,
#          0.41211282, 0.13411463, 0.12956464, 0.10567786, 0.1243872 ,
#          0.02393235, 0.03561334, 1.        , 0.34330492, 0.03202565,
#          0.96797435, 0.42820147, 0.44858363],
#         [0.16100009, 0.15240772, 0.64819684, 0.1608456 , 0.        ,
#          0.44478729, 0.13443176, 0.12899759, 0.10527648, 0.20061025,
#          0.03189981, 0.04515715, 1.        , 0.37687449, 0.03179465,
#          0.96820535, 0.4295859 , 0.44858363],
#         [0.14729102, 0.13974011, 0.64490786, 0.14714969, 0.        ,
#          0.32967459, 0.13332173, 0.12812425, 0.10465362, 0.19910279,
#          0.03310114, 0.04822538, 1.        , 0.38819889, 0.03615315,
#          0.96384685, 0.43011519, 0.44858363],
#         [0.14491268, 0.13912683, 0.64624164, 0.14477363, 0.        ,
#          0.31239778, 0.13207505, 0.12721312, 0.10399967, 0.20979212,
#          0.0342901 , 0.05348817, 1.        , 0.41685267, 0.03843934,
#          0.96156066, 0.43064448, 0.44858363],
#         [0.15004273, 0.14547413, 0.647368  , 0.14989876, 0.        ,
#          0.36988258, 0.13146987, 0.12643885, 0.10343925, 0.25564843,
#          0.03903131, 0.06019109, 1.        , 0.46105637, 0.04072553,
#          0.95927447, 0.43922396, 0.44858363],
#         [0.15297851, 0.14554669, 0.64695067, 0.15283172, 0.        ,
#          0.40156434, 0.13122148, 0.12574791, 0.10293481, 0.32702643,
#          0.04522127, 0.06726046, 1.        , 0.50769099, 0.04246554,
#          0.95753446, 0.44780345, 0.44858363],
#         [0.14756937, 0.14395902, 0.6470678 , 0.14742777, 0.        ,
#          0.35611673, 0.13044555, 0.12494537, 0.10234758, 0.37237777,
#          0.04960124, 0.07496911, 1.        , 0.53206632, 0.04025837,
#          0.95974163, 0.43975325, 0.44858363],
#         [0.15508205, 0.14642977, 0.64950645, 0.15493324, 0.        ,
#          0.43571161, 0.13050906, 0.12433272, 0.1018924 , 0.48837339,
#          0.05745716, 0.08619134, 1.        , 0.55584036, 0.03905157,
#          0.96094843, 0.44087039, 0.44858363],
#         [0.15562623, 0.15370853, 0.64990473, 0.1554769 , 0.        ,
#          0.44126116, 0.13062198, 0.12374481, 0.10145226, 0.62998584,
#          0.06694407, 0.09700064, 1.        , 0.56405368, 0.03633214,
#          0.96366786, 0.44820379, 0.44858363],
#         [0.16813661, 0.16049343, 0.65072483, 0.16797527, 0.        ,
#          0.55622941, 0.13199896, 0.12345829, 0.10122491, 0.75314762,
#          0.09447185, 0.10984716, 1.        , 0.57462118, 0.03598944,
#          0.96401056, 0.45735117, 0.44858363],
#         [0.15646967, 0.14731417, 0.64957281, 0.15631953, 0.        ,
#          0.45044281, 0.13205593, 0.12290723, 0.10080756, 0.79458179,
#          0.11065274, 0.1163758 , 1.        , 0.54850725, 0.03564675,
#          0.96435325, 0.44930097, 0.44858363],
#         [0.1520688 , 0.1414954 , 0.64689951, 0.15192288, 0.        ,
#          0.41523382, 0.13165903, 0.12226515, 0.10032293, 0.75798535,
#          0.12014357, 0.13277545, 1.        , 0.54208608, 0.03530406,
#          0.96469594, 0.44125077, 0.44858363],
#         [0.15163178, 0.14589974, 0.64755857, 0.15148628, 0.        ,
#          0.41169707, 0.13125539, 0.12162566, 0.09983749, 0.7604116 ,
#          0.1334365 , 0.15891932, 1.        , 0.55710018, 0.03830733,
#          0.96169267, 0.43651856, 0.44858363],
#         [0.15131722, 0.15034889, 0.64835844, 0.15117202, 0.        ,
#          0.40899542, 0.13085814, 0.12099155, 0.09935328, 0.76368058,
#          0.14663148, 0.19672254, 1.        , 0.55081983, 0.04090949,
#          0.95909051, 0.43222146, 0.44858363],
#         [0.15955199, 0.15712086, 0.64962444, 0.15939889, 0.        ,
#          0.49045872, 0.13133785, 0.12056073, 0.09901172, 0.74408052,
#          0.17595114, 0.25526563, 1.        , 0.54375921, 0.04187941,
#          0.95812059, 0.43348677, 0.44858363],
#         [0.15392443, 0.14323662, 0.64667274, 0.15377673, 0.        ,
#          0.43868172, 0.13119842, 0.12000809, 0.09858157, 0.67513768,
#          0.19306678, 0.3093574 , 1.        , 0.5368302 , 0.04386553,
#          0.95613447, 0.42543658, 0.44858363],
#         [0.15516979, 0.1572821 , 0.65072483, 0.1550209 , 0.        ,
#          0.45112037, 0.13119917, 0.11949524, 0.09817773, 0.61560051,
#          0.21439387, 0.43356362, 1.        , 0.53893435, 0.04126212,
#          0.95873788, 0.43473192, 0.44858363],
#         [0.16018746, 0.16780669, 0.65278725, 0.16003375, 0.        ,
#          0.50042693, 0.13171116, 0.11910876, 0.09786225, 0.62210685,
#          0.24394952, 0.66747324, 1.        , 0.53904683, 0.03865872,
#          0.96134128, 0.44404744, 0.44858363],
#         [0.17789192, 0.17095296, 0.65991265, 0.17772122, 0.        ,
#          0.63954843, 0.13397848, 0.11913999, 0.09784383, 0.70795992,
#          0.32472456, 1.        , 1.        , 0.54633565, 0.0405588 ,
#          0.9594412 , 0.44715816, 0.44858363],
#         [0.17623398, 0.16537869, 0.65603892, 0.17606488, 0.        ,
#          0.62214217, 0.13586092, 0.1191322 , 0.09779823, 0.76866962,
#          0.40832889, 1.        , 1.        , 0.5540622 , 0.04245889,
#          0.95754111, 0.4398335 , 0.43175725]]]

# In[16]:


y_prediction=model.predict(versuch1)


# 

# In[ ]:


y_prediction


# In[ ]:


y_pred


# In[ ]:




