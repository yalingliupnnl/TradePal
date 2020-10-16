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
from tradepal.src.util import genearte_input_sequence
from tradepal.src.indicators import get_XY_data
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
        best_para=pd.read_csv(filename[0], sep=',', index_col=1,delimiter=None, header='infer') 
        best_lr=best_para.loc[symbol,'ln_rate']
        
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
        best_para=pd.read_csv(filename[0], sep=',', index_col=1,delimiter=None, header='infer') 
        best_lr=best_para.loc[symbol,'ln_rate']
        best_lookback=best_para.loc[symbol,'lookback']
        
        model.compile(optimizer= Adam(lr=best_lr), loss= 'binary_crossentropy', metrics=['acc'])    
    
    #get X_past input
    df_dataX, df_dataY, df_indicators=get_XY_data(symbol, sd=dt.datetime(2020,9,1), 
                                                  ed=dt.datetime.now().date(),impact=0.0,recent_flag=True)#
    if mod!='lstm':
        X_past=df_indicators.loc[df_indicators.index[-1]].values
        X_past=X_past.reshape(1,(len(X_past)))
#                X_past=X_past[0]
    else:
        X_past=df_indicators.values[-(best_lookback+1):]
    
        
    #make prediction for today's trading option
    if mod !='lstm':
        y_today = model.predict(X_past)
    else:
        # split test set into samples
        X_ts=genearte_input_sequence(X_past, best_lookback)
        y_today = model.predict(X_ts, verbose=0)[-1][0].astype(int)

    
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
              'SVM','lstm']#'DNN'
    names=["LogReg","RandForest","AdaBoosting",
               "SVM",
               "LSTM"]
    #results save the prediction from each fund with each ML method
    results = pd.DataFrame(columns=models,data=None,index=symbols)
    backtest_acc= pd.DataFrame(columns=models,data=None,index=symbols)
    recommendation=pd.DataFrame(columns=['trade'],data=None,index=symbols)
    
    for symbol in symbols:
        #get X_past input
        df_dataX, df_dataY, df_indicators=get_XY_data(symbol, sd=dt.datetime.now().date()-dt.timedelta(1115), 
                                                  ed=dt.datetime.now().date(),impact=0.0,recent_flag=True)#
        
        #prediction results for the the past nn (750) transaction days, the prediction on 10/1 corresponds
        #to actual optimal trading on 10/2, so the index of results (trading option) here shift 1 day forward 
        #just for the sake of plotting
        nn=750
        results_past=pd.DataFrame(columns=models,data=None,index=df_indicators.index[-(nn+1):-1])
        
        labelencoder = LabelEncoder()
        df_dataY['encode']=labelencoder.fit_transform(df_dataY['dataY'])
        y_true=df_dataY['encode']
        
        #the last 2 days are omitted
        #e.g., for day 1-24, features of day 22 is used to predict actual optimal trading on day 23 (Y-label),
        # which is determined by price on day 24, so day 23 and 24 are omited in Y label.
        idx1=y_true.index.get_loc(df_indicators.index[-(nn+2)].date().isoformat())
        
        results_past['y_true']=y_true[idx1:].values.reshape(nn,1)
        for mod in models:
            results.loc[symbol,mod], model = recommend(symbol,mod)
            
            #get the prediction resutls of the past nn(750) transaction days
            if mod!='lstm':
                X_past=df_indicators.iloc[-(nn+2):-2,:].values

            else:
                filename=glob.glob('tradepal/models/lstm_best_para.csv')
                best_para=pd.read_csv(filename[0], sep=',', index_col=1,delimiter=None, header='infer')                 
                best_lookback=best_para.loc[symbol,'lookback']
                X_past=df_indicators.iloc[-(nn+best_lookback+2):-2,:].values          
                
            #make prediction for today's trading option
            if mod !='lstm':
                y_past = model.predict(X_past)
            else:
                # split test set into samples
                X_ts=genearte_input_sequence(X_past, best_lookback)
                y_past = model.predict(X_ts, verbose=0).astype(int)

            y_option=np.zeros((len(y_past),1)).astype(int)
            if mod not in ['lstm']:   
                for i in np.arange(len(y_past)):
                    y_option[i]=y_past[i].astype(int) 
            elif mod=='lstm':
                y_option=y_past
            
            results_past[mod]=y_option
            print(dt.datetime.now())
        results_past.to_csv('tradepal/models/results_past_'+symbol+'.csv', sep=',')
        diff=results_past[models].values-np.tile(
                results_past['y_true'].values.reshape(len(results_past.index),1),(1,len(models)))
        diff[diff==0]=100
        diff[diff!=100]=-1
        diff[diff==100]=1
        perform=pd.DataFrame(columns=models,data=diff,index=results_past.index)
        
        #plot the map out
        fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)    
        cax = ax.matshow(perform.T, interpolation=None, aspect='auto', cmap='RdYlGn')

        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['-1', '0', '1'], fontsize=16)  # vertically oriented colorbar
        ax.set_xticks([55,180,306,431,558,684])
        
        # ... and label them with the respective list entries
        ax.set_xticklabels(['1/2/18','7/2/18','1/2/19','7/2/19','1/2/20','7/2/20'], fontsize=16)
        ax.set_yticklabels(['']+names, fontsize=16)
        ax.xaxis.set_tick_params(width=5)
        ax.yaxis.set_tick_params(width=5)
        fig.tight_layout()
#        plt.show()
        plt.savefig("tradepal/resources/back_test_model_performance_"+symbol+'.png')
        
        #plot the bar chart of accuracy for each model
        diff[diff!=1]=0
        acc=diff.sum(axis=0)/diff.shape[0]
        backtest_acc.loc[symbol]=acc
        
        fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)        
        plt.style.use('ggplot')        
        x_pos = [i for i, _ in enumerate(names)]        
        ax.bar(x_pos, acc, color='green',width=0.5)
        ax.set_xlabel("models",fontsize=15)
        ax.set_ylabel("accuracy",fontsize=15)
        ax.set_title("backtesting prediction accuracy for all models in the past 3 years",fontsize=15)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=15)
        ax.set_ylim([0.5,0.88])
        ax.set_yticks([0.5,0.6,0.7,0.8])
        ax.set_yticklabels(["0.5","0.6","0.7","0.8"], fontsize=15)
        fig.tight_layout()        
#        plt.show()
        plt.savefig("tradepal/resources/back_test_performance_barchart_"+symbol+'.png')
        
    backtest_acc.to_csv('tradepal/models/backtest_accuracy.csv', sep=',')    
        
    for i in np.arange(len(symbols)):
        final=results['AdaBoostClassifier'].values
        if final[i]==0:
            recommendation['trade'][i]='BUY'
        elif final[i]==1:
            recommendation['trade'][i]='HOLD'
        elif final[i]==2:
            recommendation['trade'][i]='SELL'
    query=results 
    query['recommend']=recommendation['trade']     
    query.to_csv('tradepal/models/query.csv', sep=',')
    return recommendation, results, query


if __name__ == "__main__":
    recommend_all()
    