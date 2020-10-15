#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:03:09 2020

@author: yalingliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 08:27:36 2020

@author: yalingliu
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:42:25 2019

@author: User1
"""
import glob
import pandas as pd
import datetime as dt
#import keras
# from model.my_preprocess_noEVI_noID import get_ds
# from model.utils import r2, split_sequence, calculate_aic
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Dropout, Masking,Input, Activation#.python
from tensorflow.keras.models import Sequential, Model
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tradepal.util import calculate_aic, r2, split_sequence
from tradepal.indicators import get_XY_data


def lstm(lstm_params,symbol = "SPY", sd=dt.datetime(1993,1,29), ed=dt.datetime(2020,8,31), impact=0.0):
    df_dataX, df_dataY, df_indicators=get_XY_data(symbol, sd, ed, impact,recent_flag=False)
        
    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    df_dataY['encode']=labelencoder.fit_transform(df_dataY['dataY'])
    
#    labels = pd.get_dummies(df_dataY['dataY'], prefix='option')#one hot encode
    
    X = df_dataX.values
    # y = enc_df.values
    y=df_dataY['encode'].values
#    y=labels.values
    
    
    # split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # pre-process training data, fit the scaler on training data only,
    # then standardise both training and test sets with the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_ds=np.column_stack((X_train,y_train))
    test_ds=np.column_stack((X_test,y_test))
     
    # split into samples
    X_tr, y_tr = split_sequence(train_ds, lstm_params['lookback'])
    y_tr = y_tr[:,-1:]
    y_tr =y_tr.astype(int)
    X_tr=X_tr[:,:,:-1]
   
    # define model    
    model = Sequential()
    model.add(LSTM(lstm_params['n_units'], return_sequences=True, input_shape=(lstm_params['lookback'], X_tr.shape[2])))
#    model.add(LSTM(lstm_params['n_units'], return_sequences=True,activity_regularizer = l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    #output layer
    model.add(Dense(y_tr.shape[1], activation='softmax')) 
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lstm_params['ln_rate']), metrics=['acc'])
    #sparse_categorical_crossentropy,binary_crossentropy
    
    
     # define model, good to go    
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(lstm_params['lookback'], X_tr.shape[2])))
    model.add(LSTM(lstm_params['n_units'], activity_regularizer = l2(0.001)))
#    model.add(LSTM(lstm_params['n_units'], return_sequences=True,activity_regularizer = l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(y_tr.shape[1], activation='softmax')) 
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lstm_params['ln_rate']), metrics=['acc'])
##    print(model.summary())
   
    #fit the model
    history = model.fit(X_tr, y_tr,
                        batch_size= lstm_params['batch_size'], epochs=15, verbose=0,  
                        validation_split=0.1)
    #make train predictions
#    train_ypred = model.predict(X_tr, verbose=0)
   
   
    # split test set into samples
    X_ts, y_ts = split_sequence(test_ds, lstm_params['lookback'])
    y_ts = y_ts[:,-1:]
    X_ts=X_ts[:,:,:-1]
    #make test predictions
#    test_ypred = model.predict(X_ts, verbose=0)
    
    #evaluate accuray
    score = model.evaluate(X_ts, y_ts, verbose=0)
#    print("Test Score:", score[0])
#    print("Test Accuracy:", score[1])#0.738, 1% for 2-day return
    
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc   = history.history['val_acc']    
    epoch_idx  = np.arange(1, 20+1)
   
    model_loss = {'epoch_idx': epoch_idx,
                  'train_loss':train_loss,
                  'val_loss':val_loss,
                  'train_acc':train_acc,
                  'val_acc':val_acc,   
                  'test_acc':score[1]
                  } 
   
    return model, model_loss

if __name__ == "__main__":
    symbols = ["SPY","DIA","QQQ","TLT","IWM"]
    filename=glob.glob('tradepal/models/lstm_best_para.csv')
    best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
    best_para=best_para.set_index('symbol')
    
    for sym in symbols:        
        lstm_params = {
                    'lookback':best_para.loc[sym,'lookback'].astype(int),
                   'n_units':best_para.loc[sym,'n_units'].astype(int),
                   'batch_size': best_para.loc[sym,'batch_size'].astype(int),#4500
                   'ln_rate' : best_para.loc[sym,'ln_rate']} #0.05
        #different index funds have different start date
        if sym=='SPY':
            sd=dt.datetime(1993,1,29)
        elif sym=='DIA':
            sd=dt.datetime(1998,1,20)
        elif sym=='QQQ':
            sd=dt.datetime(1999,3,10)
        elif sym=='TLT':
            sd=dt.datetime(2002,7,30)
        elif sym=='IWM':
            sd=dt.datetime(2000,5,26)
     
        #run the model
        start_time=dt.datetime.now()
        model, model_loss = lstm(lstm_params,symbol = sym, sd=sd, 
                           ed=dt.datetime(2020,8,31), impact=0.0)  
        model_loss['test_acc']
         # serialize model to JSON
        name='tradepal/models/lstm_'+sym
        model_json = model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name+ ".h5")
        print("Saved lstm model to disk for "+sym+"!")
        
        end_time=dt.datetime.now()
        use_time=end_time - start_time
        use_minutes=(use_time.seconds+use_time.microseconds/1000000)/60.0  
        print('time used: '+str(use_minutes)+ ' minutes!')