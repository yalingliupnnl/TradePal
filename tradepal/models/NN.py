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
from tradepal.util import calculate_aic, r2
from indicators import get_XY_data


def myANN(ANN_params,symbol = "SPY", sd=dt.datetime(1993,1,29),ed=dt.datetime.now().date(), impact=0.0):
    df_dataX, df_dataY, df_indicators=get_XY_data(symbol, sd, ed, impact)
        
    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    df_dataY['encode']=labelencoder.fit_transform(df_dataY['dataY'])
    
    labels = pd.get_dummies(df_dataY['dataY'], prefix='option')
    
    #one hot encoder for the trading options of buy, hold and sell
#    enc = OneHotEncoder(handle_unknown='ignore')
#    enc_df = pd.DataFrame(enc.fit_transform(df_dataY[['encode']]).toarray())
    # inv_df=enc.inverse_transform(enc_df)
    
    X = df_dataX.values
    # y = enc_df.values
    # y=df_dataY['encode'].values
    y=labels.values
    
    # split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # pre-process training data, fit the scaler on training data only,
    # then standardise both training and test sets with the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
        ######define model  
    input_layer = Input(shape=(X.shape[1],))
    dense_layer_1 = Dense(ANN_params['n_units'], activation='relu')(input_layer)
    dense_layer_2 = Dense(ANN_params['n_units'], activation='relu')(dense_layer_1)
    output = Dense(y.shape[1], activation='softmax')(dense_layer_2)
    
    model = Model(inputs=input_layer, outputs=output)
    optimizer = Adam(lr=ANN_params['ln_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])    
#    print(model.summary())
    
     #fit the model
#    print('starte fitting the ANN model')
#    history = model.fit(X_train, y_train, batch_size=500, epochs=20, verbose=1, validation_split=0.1)
    history = model.fit(X_train, y_train,batch_size= ANN_params['batch_size'],
                     epochs=20, verbose=0, validation_split=0.1)
    
     #make train predictions
#    train_pred=model.predict(X_train)
#    np.argmax(train_pred[0])
#    y_test[0]
    
    score = model.evaluate(X_test, y_test, verbose=0)
    
#    print("Test Score:", score[0])
#    print("Test Accuracy:", score[1])
    #0.757 for 1% 2-day return, 
    #0.872 for 1.5% 2-day return,
    #0.782 for 1.5% 3-day return, 
    #0.9258 for 2% 2-day return
    #decrease time span for return, accuracy increase
    #increase threshold for return, accuracy increase
        
   
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
   
    #make test predictions
#    test_ypred = model.predict(X_test, verbose=0)    
    return  model, model_loss

if __name__ == "__main__":
    symbols = ["SPY","DIA","QQQ","TLT","IWM"]
    filename=glob.glob('tradepal/models/ANN_best_para.csv')
    best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
    best_para=best_para.set_index('symbol')

    for sym in symbols:        
        ANN_params = {
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
        model, model_loss = myANN(ANN_params,symbol = sym, sd=sd, 
                           ed=dt.datetime(2020,8,31), impact=0.0)  
        model_loss['test_acc']
         # serialize model to JSON
        name='tradepal/models/DNN_'+sym
        model_json = model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name+ ".h5")
        print("Saved DNN model to disk for "+sym+"!")
        
        end_time=dt.datetime.now()
        use_time=end_time - start_time
        use_minutes=(use_time.seconds+use_time.microseconds/1000000)/60.0  
        print('time used: '+str(use_minutes)+ ' minutes!')
        
#    ANN_params = {'hid_layers': best_para['hid_layers'][0],#2
#               'n_units':best_para['n_units'][0],#64
#               'batch_size': best_para['batch_size'][0],#500
#               'ln_rate' : best_para['ln_rate'][0]} #0.005  
    
    # define model    
#    model = Sequential()
#    for i in np.arange(ANN_params['hid_layers']-1):
#        # Add fully connected layer with a LeakyReLU activation function
#        model.add(Dense(units=ANN_params['n_units'], activation='relu',input_shape=(X.shape[1],)))#, activity_regularizer = l2(0.001)
#    #    model.add(Dropout(0.2))  
##        model.add(LeakyReLU(alpha=0.1))
#       
#    # Add fully connected layer with no activation function
#    model.add(Dense(y.shape[1], activation='softmax'))
#    optimizer = Adam(lr=ANN_params['ln_rate'])  #lr=0.0005, RMSProp  
#    model.compile(optimizer= optimizer, loss= 'categorical_crossentropy', metrics=['acc'])
##    print(model.summary())
    
    