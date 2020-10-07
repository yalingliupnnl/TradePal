#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:19:03 2020

@author: yalingliu
"""

import pandas as pd
import datetime as dt
import numpy as np
import pickle
import glob
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tradepal.util import genearte_input_sequence
from indicators import get_XY_data
from sklearn.preprocessing import LabelEncoder
#from tradepal.models import logReg,randForest,adaBst,SVM_model

def recommend(symbol,mod):
    # load the model from disk
    if mod=='LogisticRegression':    
        filename = 'tradepal/models/LogisticRegression_' + symbol+ '.sav'        
        model = pickle.load(open(filename, 'rb'))
    elif mod=='RandomForestClassifier':    
        filename = 'tradepal/models/RandomForestClassifier_' + symbol+ '.sav'        
        model = pickle.load(open(filename, 'rb'))    
    elif mod=='AdaBoostClassifier':    
        filename = 'tradepal/models/AdaBoostClassifier_' + symbol+ '.sav'        
        model = pickle.load(open(filename, 'rb'))  
    elif mod=='SVM':    
        filename = 'tradepal/models/SVM_' + symbol+ '.sav'        
        model = pickle.load(open(filename, 'rb'))  
    elif mod=='DNN':    
        name='tradepal/models/DNN_'+symbol  
        json_file = open(name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(name+".h5")
        
        filename=glob.glob('tradepal/models/ANN_best_para.csv')
        best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
        if symbol=="SPY":
            best_lr=best_para.loc[0,'ln_rate']
        elif symbol=="DIA":
            best_lr=best_para.loc[1,'ln_rate']
        elif symbol=="QQQ":
            best_lr=best_para.loc[2,'ln_rate']
        elif symbol=="TLT":
            best_lr=best_para.loc[3,'ln_rate']
        elif symbol=="IWM":
            best_lr=best_para.loc[4,'ln_rate']
        
        model.compile(optimizer= Adam(lr=best_lr), loss= 'categorical_crossentropy', metrics=['acc'])       
        
    elif mod =='lstm':    
        name='tradepal/models/lstm_'+symbol  
        json_file = open(name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(name+".h5")
        
        filename=glob.glob('tradepal/models/lstm_best_para.csv')
        best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
        if symbol=="SPY":
            best_lr=best_para.loc[0,'ln_rate']
        elif symbol=="DIA":
            best_lr=best_para.loc[1,'ln_rate']
        elif symbol=="QQQ":
            best_lr=best_para.loc[2,'ln_rate']
        elif symbol=="TLT":
            best_lr=best_para.loc[3,'ln_rate']
        elif symbol=="IWM":
            best_lr=best_para.loc[4,'ln_rate']
        
        model.compile(optimizer= Adam(lr=best_lr), loss= 'binary_crossentropy', metrics=['acc'])    
    
    #get X_past input
    df_dataX, df_dataY, df_indicators=get_XY_data(symbol, sd=dt.datetime(2020,9,1), 
                                                  ed=dt.datetime.now().date(),impact=0.0,recent_flag=True)#
    if mod!='lstm':
        X_past=df_indicators.loc[df_indicators.index[-1]].values
        X_past=X_past.reshape(1,(len(X_past)))
#                X_past=X_past[0]
    else:
        X_past=df_indicators.values[-6:]
    
        
    #make prediction for today's trading option
    if mod !='lstm':
        y_today = model.predict(X_past)
    else:
        # split test set into samples
        X_ts=genearte_input_sequence(X_past, best_para['lookback'][0])
        y_today = model.predict(X_ts, verbose=0)[-1][0].astype(int)
#                y_ts = y_ts[:,-1:]
#                X_ts=X_ts[:,:,:-1]
    
    if mod not in ['DNN','lstm']:            
        y_option=y_today[0].astype(int) 
    elif mod =='DNN':
        y_option=np.argmax(y_today)
    elif mod =='lstm':
        y_option=y_today
    
    return y_option, model #0,1,2(BUY,HOLD,SELL)
            

def recommend_all():
    symbols = ["SPY","DIA","QQQ","TLT","IWM"]    
    models = ['LogisticRegression','RandomForestClassifier','AdaBoostClassifier',
              'SVM','DNN','lstm']
    #results save the prediction from each fund with each ML method
    results = pd.DataFrame(columns=models,data=None,index=symbols)
    recommendation=pd.DataFrame(columns=['trade'],data=None,index=symbols)
    
    for symbol in symbols:
        #get X_past input
        df_dataX, df_dataY, df_indicators=get_XY_data(symbol, sd=dt.datetime(2020,9,1), 
                                                  ed=dt.datetime.now().date(),impact=0.0,recent_flag=True)#
        
        #prediction results for the the past 22 transaction days, the prediction on 10/1 corresponds
        #to actual optimal trading on 10/2, so the index here shift 1 day forward just for the sake of plotting
        nn=247
        results_past=pd.DataFrame(columns=models,data=None,index=df_indicators.index[-(nn+1):-1])
        
        labelencoder = LabelEncoder()
        df_dataY['encode']=labelencoder.fit_transform(df_dataY['dataY'])
        y_true=df_dataY['encode']
        #start from -24th day to -3 day, because the last 2 days are omitted
        #e.g., for day 1-24, features of day 22 is used to predict optimal trading on day 23 (Y-label),
        # which is determined by price on day 24, so day 23 and 24 are omited in Y label.
        idx1=y_true.index.get_loc(df_indicators.index[-(nn+2)].isoformat())
#        idx2=y_true.index.get_loc(results_past.index[-1].isoformat())
        
        results_past['y_true']=y_true[idx1:].values.reshape(nn,1)
        for mod in models:
            results.loc[symbol,mod], model = recommend(symbol,mod)
            
            #get the prediction resutls of the past 22 transaction days
            if mod!='lstm':
#                idx=df_indicators.index.get_loc((dt.datetime.now().date()-dt.timedelta(30)).isoformat())
                X_past=df_indicators.iloc[-(nn+2):-2,:].values
#                X_past=X_past.reshape(1,(len(X_past)))
            else:
                X_past=df_indicators.iloc[-(nn+7):-2,:].values
#            
                
            #make prediction for today's trading option
            if mod !='lstm':
                y_past = model.predict(X_past)
            else:
                filename=glob.glob('tradepal/models/lstm_best_para.csv')
                best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer')
                # split test set into samples
                X_ts=genearte_input_sequence(X_past, best_para['lookback'][0])
                y_past = model.predict(X_ts, verbose=0).astype(int)

            y_option=np.zeros((len(y_past),1)).astype(int)
            if mod not in ['DNN','lstm']:   
                for i in np.arange(len(y_past)):
                    y_option[i]=y_past[i].astype(int) 
            elif mod =='DNN':
                for i in np.arange(len(y_past)):
                    y_option[i]=np.argmax(y_past[i])
            elif mod=='lstm':
                y_option=y_past
            
            results_past[mod]=y_option
            print(dt.datetime.now())
#        pd.set_option('display.expand_frame_repr', False)
    
     
    #final recommendation is based on DNN prediction due to its overall better performance
    for i in np.arange(len(symbols)):
        final=results['DNN'].values
        if final[i]==0:
            recommendation['trade'][i]='BUY'
        elif final[i]==1:
            recommendation['trade'][i]='HOLD'
        elif final[i]==2:
            recommendation['trade'][i]='SELL'
            
    return recommendation, results



def recommend_all_2():
    symbols = ["SPY","DIA","QQQ","TLT","IWM"]
#    filename=glob.glob('model_score_by_symbol.csv')
#    model_score=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
#    model_score.set_index('symbol')#set the sybol as index
    
    models = ['LogisticRegression','RandomForestClassifier','AdaBoostClassifier',
              'SVM','DNN','lstm']
    #results save the prediction from each fund with each ML method
    results = pd.DataFrame(columns=models,data=None,index=symbols)
    recommendation=pd.DataFrame(columns=['trade'],data=None,index=symbols)
    for symbol in symbols:
        for mod in models:
            # load the model from disk
            if mod=='LogisticRegression':    
                filename = 'tradepal/models/LogisticRegression_' + symbol+ '.sav'        
                model = pickle.load(open(filename, 'rb'))
            elif mod=='RandomForestClassifier':    
                filename = 'tradepal/models/RandomForestClassifier_' + symbol+ '.sav'        
                model = pickle.load(open(filename, 'rb'))    
            elif mod=='AdaBoostClassifier':    
                filename = 'tradepal/models/AdaBoostClassifier_' + symbol+ '.sav'        
                model = pickle.load(open(filename, 'rb'))  
            elif mod=='SVM':    
                filename = 'tradepal/models/SVM_' + symbol+ '.sav'        
                model = pickle.load(open(filename, 'rb'))  
            elif mod=='DNN':    
                name='tradepal/models/DNN_'+symbol  
                json_file = open(name+'.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                # load weights into new model
                model.load_weights(name+".h5")
                
                filename=glob.glob('tradepal/models/ANN_best_para.csv')
                best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
                if symbol=="SPY":
                    best_lr=best_para.loc[0,'ln_rate']
                elif symbol=="DIA":
                    best_lr=best_para.loc[1,'ln_rate']
                elif symbol=="QQQ":
                    best_lr=best_para.loc[2,'ln_rate']
                elif symbol=="TLT":
                    best_lr=best_para.loc[3,'ln_rate']
                elif symbol=="IWM":
                    best_lr=best_para.loc[4,'ln_rate']
                
                model.compile(optimizer= Adam(lr=best_lr), loss= 'categorical_crossentropy', metrics=['acc'])       
                
            elif mod =='lstm':    
                name='tradepal/models/lstm_'+symbol  
                json_file = open(name+'.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                # load weights into new model
                model.load_weights(name+".h5")
                
                filename=glob.glob('tradepal/models/lstm_best_para.csv')
                best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
                if symbol=="SPY":
                    best_lr=best_para.loc[0,'ln_rate']
                elif symbol=="DIA":
                    best_lr=best_para.loc[1,'ln_rate']
                elif symbol=="QQQ":
                    best_lr=best_para.loc[2,'ln_rate']
                elif symbol=="TLT":
                    best_lr=best_para.loc[3,'ln_rate']
                elif symbol=="IWM":
                    best_lr=best_para.loc[4,'ln_rate']
                
                model.compile(optimizer= Adam(lr=best_lr), loss= 'binary_crossentropy', metrics=['acc'])    
            
            #get X_past input
            df_dataX, df_dataY, df_indicators=get_XY_data(symbol, sd=dt.datetime(2020,9,1), 
                                                          ed=dt.datetime.now().date(),impact=0.0,recent_flag=True)#
            if mod!='lstm':
                X_past=df_indicators.loc[df_indicators.index[-1]].values
                X_past=X_past.reshape(1,(len(X_past)))
#                X_past=X_past[0]
            else:
                X_past=df_indicators.values[-6:]
            
                
            #make prediction for today's trading option
            if mod !='lstm':
                y_today = model.predict(X_past)
            else:
                # split test set into samples
                X_ts=genearte_input_sequence(X_past, best_para['lookback'][0])
                y_today = model.predict(X_ts, verbose=0)[-1][0].astype(int)
#                y_ts = y_ts[:,-1:]
#                X_ts=X_ts[:,:,:-1]
            
            if mod not in ['DNN','lstm']:            
                y_option=y_today[0].astype(int) 
            elif mod =='DNN':
                y_option=np.argmax(y_today)#0,1,2(BUY,HOLD,SELL)
            elif mod == 'lstm':
                y_option=y_today
            results.loc[symbol,mod]=y_option
            
#        print('Finished predicting for '+ symbol+', keep going!')
#        print(dt.datetime.now())
    
#if there are ties, choose the prediction from DNN, otherwise choose the one that has most votes
#    BUY_score=np.sum(results.values==0,axis=1)
#    HOLD_score=np.sum(results.values==1,axis=1)
#    SELL_score=np.sum(results.values==2,axis=1)
#    score=np.column_stack((BUY_score,HOLD_score,SELL_score))#5*3 array 
#    final=[]
    
    #final recommendation is based on DNN prediction due to its overall better performance
    for i in np.arange(len(symbols)):
#        if score[i,0]==score[i,1] or score[i,0]==score[i,2] or score[i,1]==score[i,2]:
#            final.append(results.iloc[i,'DNN'])
#        else:
#            final.append(np.argmax(score,axis=1))
        final=results['DNN'].values
        if final[i]==0:
            recommendation['trade'][i]='BUY'
        elif final[i]==1:
            recommendation['trade'][i]='HOLD'
        elif final[i]==2:
            recommendation['trade'][i]='SELL'
            
    return recommendation, results
