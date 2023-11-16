

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta as ta
from pandas import Series
from pandas_ta.overlap import linreg
from pandas_ta.utils import get_offset, verify_series


# In[2]:




# In[3]:
def nz(value,default):
        return default if pd.isnull(value) else value
def na(value):
        return pd.isnull(value)
    
                        ################## Halftrend Indicator#########
def halftrend(High, Low, Open,Close,amplitude=None, channelDeviation=None,**kwargs):
    
    out=[]
    trend=0
    nextTrend=0
    up=0.0
    down=0.0
    atrHigh=0.0
    atrLow=0.0
    direction=None
    atrlen=14
    amplitude=2
    channel_deviation=2
    df = {
    'High': High,  # High sütunu verileri
    'Low': Low,    # Low sütunu verileri
    'Open': Open,  # Open spalte
    'Close': Close  # Close cloumn
    }

    data = pd.DataFrame(df)
    atr_N=ta.atr(data.High,data.Low,data.Close,window=atrlen)
    highma_N=ta.sma(data.High,amplitude)
    lowma_N=ta.sma(data.Low,amplitude)
    highestbars=data.High.rolling(amplitude,min_periods=1).max()
    lowestbars = data['Low'].rolling(amplitude, min_periods=1).min()
    data['highestbars'] = highestbars
    data['lowestbars'] = lowestbars
    arrTrend=[None]*len(data)
    
    
    arrUp=[None]*len(data)
    arrDown=[None]*len(data)

    maxLowPrice = data['Low'].iat[atrlen - 1]
    minHighPrice = data['High'].iat[atrlen - 1]

    if data['Close'].iat[0] > data['Low'].iat[atrlen]:
        trend = 1
        nextTrend = 1

    df=data.copy()
    for i in range(1, len(data)):
        atr2 = atr_N.iat[i] / 2.0
        dev = channel_deviation * atr2

        highPrice = highestbars.iat[i]
        lowPrice = lowestbars.iat[i]


        if nextTrend == 1:
            maxLowPrice = max(lowPrice, maxLowPrice)
            if highma_N.iat[i] < maxLowPrice and df['Close'].iat[i] < nz(df['Low'].iat[i - 1], df['Low'].iat[i]):
                trend = 1
                nextTrend = 0
                minHighPrice = highPrice
        else:
            minHighPrice = min(highPrice, minHighPrice)
            if lowma_N.iat[i] > minHighPrice and df['Close'].iat[i] > nz(df['High'].iat[i - 1], df['High'].iat[i]):
                trend = 0
                nextTrend = 1
                maxLowPrice = lowPrice
        arrTrend[i] = trend

        if trend == 0:
            if not na(arrTrend[i - 1]) and arrTrend[i - 1] != 0:
                up = down if na(arrDown[i - 1]) else arrDown[i - 1]
            else:
                up = maxLowPrice if na(arrUp[i - 1]) else max(maxLowPrice, arrUp[i - 1])
            direction = 1
            atrHigh = up + dev
            atrLow = up - dev
            arrUp[i] = up
        else:
            if not na(arrTrend[i - 1]) and arrTrend[i - 1] != 1:
                down = up if na(arrUp[i - 1]) else arrUp[i - 1]
            else:
                down = minHighPrice if na(arrDown[i - 1]) else min(minHighPrice, arrDown[i - 1])
            direction = -1
            atrHigh = down + dev
            atrLow = down - dev
            arrDown[i] = down

        if trend == 0:
            out.append([atrHigh, up, atrLow, direction, arrUp[i], arrDown[i]])
        else:
            out.append([atrHigh, down, atrLow, direction, arrUp[i], arrDown[i]])
    result = pd.DataFrame(out, columns=['atrHigh', 'close', 'atrLow', 'direction', 'arrUp', 'arrDown'])
    #result.append([np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
    nan_row = pd.DataFrame([[np.nan] * 6], columns=['atrHigh', 'close', 'atrLow', 'direction', 'arrUp', 'arrDown'])
    result = pd.concat([nan_row, result]).reset_index(drop=True)
    result.reset_index()
    result.index = df.index
    #print(result)
    
    return result,df


    
###################################Average Sentiment Oscillator Indicator####################

def aso(High, Low, Open,Close, length,**kwargs):
    mode = 0
    df = {
    'High': High,  # High sütunu verileri
    'Low': Low,    # Low sütunu verileri
    'Open': Open,  # Open spalte
    'Close': Close  # Close cloumn
    }
    data = pd.DataFrame(df)
    data['intrarange'] = data['High'] - data['Low']
    data['grouplow'] = data['Low'].rolling(length).min()
    data['grouphigh'] = data['High'].rolling(length).max()
    data['groupopen'] = data['Open'].shift(length - 1)
    data['grouprange'] = data['grouphigh'] - data['grouplow']

    K1 = data['intrarange'].apply(lambda x: 1 if x == 0 else x)
    K2 = data['grouprange'].apply(lambda x: 1 if x == 0 else x)

    data['intrabarbulls'] = ((((data['Close'] - data['Low']) + (data['High'] - data['Open'])) / 2) * 100) / K1
    data['groupbulls'] = ((((data['Close'] - data['grouplow']) + (data['grouphigh'] - data['groupopen'])) / 2) * 100) / K2
    data['intrabarbears'] = ((((data['High'] - data['Close']) + (data['Open'] - data['Low'])) / 2) * 100) / K1
    data['groupbears'] = ((((data['grouphigh'] - data['Close']) + (data['groupopen'] - data['grouplow'])) / 2) * 100) / K2

    data['TempBufferBulls'] = data.apply(lambda row: (row['intrabarbulls'] + row['groupbulls']) / 2 if mode == 0 else row['intrabarbulls'] if mode == 1 else row['groupbulls'], axis=1)
    data['TempBufferBears'] = data.apply(lambda row: (row['intrabarbears'] + row['groupbears']) / 2 if mode == 0 else row['intrabarbears'] if mode == 1 else row['groupbears'], axis=1)

    data['ASOBulls'] = data['TempBufferBulls'].rolling(length).mean()
    data['ASOBears'] = data['TempBufferBears'].rolling(length).mean()
    
    data["direction"]=data.apply(lambda row: -1 if row['ASOBulls'] < row['ASOBears'] else 1, axis=1)

    data.dropna(inplace=True)
    return data.ASOBulls-data.ASOBears
# In[ ]:


###################################Trend_Intensity_Index##################################

def TII(High, Low, Open,Close,majorLength,minorLength,upperLevel,lowerLevel, **kwargs):
    
    highlightBreakouts = True
    df = {
    'High': High,  # High sütunu verileri
    'Low': Low,    # Low sütunu verileri
    'Open': Open,  # Open spalte
    'Close': Close  # Close cloumn
    }
    data = pd.DataFrame(df)
    data["SMA"] = data['Close'].rolling(window=majorLength).mean()
    data.fillna(data.SMA.iloc[majorLength - 1],inplace=True)
    
    
    signals = []

    tii_values=[]
    for i in range(len(data)): #np.arange(0,len(data),20):  drop(data.index)
        #if i % minorLength == 1:
        tii=0
        positiveSum = 0.0
        negativeSum = 0.0
        close=[]
        Sma=[]
        close=data['Close'].iloc[i:20+i].values
        Sma=data['SMA'].iloc[i:20+i].values
        for i in range(len(close)):
            price = close[i]
            avg = Sma[i]
            if price > avg:
                positiveSum += price - avg
            else:
                negativeSum += avg - price

        
        tii =100 * positiveSum / (positiveSum + negativeSum)
        tii_values.append(tii)
            # "long" veya "short" sinyal üret
        if tii > upperLevel and highlightBreakouts:
            signal = 1
        elif tii < lowerLevel and highlightBreakouts:
            signal = -1
        else:
            signal = 0
        signals.append(signal)        



    data["Signals"]=signals
    
    return tii_values

                          ##################BCL2ECTS INDICATOR##########
def BCL2ECTS(High, Low, Open,Close, fast_length,slow_length,buy_threshold,sell_threshold,**kwargs):
    
    df = {
    'High': High,  # High sütunu verileri
    'Low': Low,    # Low sütunu verileri
    'Open': Open,  # Open spalte
    'Close': Close  # Close cloumn
    }
    data = pd.DataFrame(df)
    price = data['Close']
    def cti(close, length=None, offset=None, **kwargs) -> Series:
        # """Indicator: Correlation Trend Indicator"""
        length = int(length) if length and length > 0 else 12
        close = verify_series(close, length)
        offset = get_offset(offset)

        if close is None: return

        cti = linreg(close, length=length, r=True)

        # Offset
        if offset != 0:
            cti = cti.shift(offset)

        # Handle fills
        if "fillna" in kwargs:
            cti.fillna(method=kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            cti.fillna(method=kwargs["fill_method"], inplace=True)

        cti.name = f"CTI_{length}"
        cti.category = "momentum"
        return cti
    
    corr_f = cti(price,fast_length)
    corr_s=cti(price,slow_length)


    corr_f.fillna(corr_f[19],inplace=True)
    corr_s.fillna(corr_s[39],inplace=True)
    data["CorrF"], data["CorrS"]=corr_f,corr_s
    
    buy_signals = (data.CorrF > data.CorrS ) & (data.CorrF < data.CorrS.shift(1))
    buy_signals = buy_signals.replace(True, 1)
    buy_signals = buy_signals.replace(False, 0)
    #sell_signals = (np.array(corr_f) < sell_threshold)
    sell_signals = (data.CorrF < data.CorrS ) & (data.CorrF > data.CorrS.shift(1))
    sell_signals = sell_signals.replace(True, -1)
    sell_signals = sell_signals.replace(False, 0)
    data["Buy_Signals"],data["Sell_Signals"]=buy_signals,sell_signals

    signals = []
    for i in range(len(buy_signals)):
        if buy_signals[i]:
            signals.append(1)
        elif sell_signals[i]:
            signals.append(-1)
        else:
            signals.append(0)

    data["Signals"]=signals
    
    return data["CorrF"], data["CorrS"]
 
    
                    ######################### THE PRICE RADIO ################
    
def TPR(High, Low, Open,Close,length,**kwargs):
    
    def clamp(value, minimum, maximum):
        t = minimum if value < minimum else value
        t = maximum if t > maximum else t
        return t

    def am(signal, period):
        envelope = signal.abs().rolling(4).max()
        return ta.sma(envelope,length=period)
    
    df = {
    'High': High,  # High sütunu verileri
    'Low': Low,    # Low sütunu verileri
    'Open': Open,  # Open spalte
    'Close': Close  # Close cloumn
    }
    data = pd.DataFrame(df)
    
    # Difference between current value and previous, x - x[y].
    data['deriv'] = data['Close'].diff()
    data['deriv'].iat[0] = data.deriv.iat[1]

    data.deriv

    #the AM+ and AM- Signals
    data['AM+'] = am(data.deriv,length)
    data['AM-'] = -am(data.deriv,length)

    data["FM"]=data.deriv*10
    
    data["l"]=data.deriv.rolling(window=length).min()
    data["h"]=data.deriv.rolling(window=length).max()

    data["FM"]=ta.sma(data.apply(lambda row: clamp(row["FM"],row["l"],row["h"]),axis=1),length=length)
    
    return data["AM+"],data["AM-"],data.FM

    




