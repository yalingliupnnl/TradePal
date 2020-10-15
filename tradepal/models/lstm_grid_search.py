#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:22:57 2020

@author: yalingliu
"""

import pandas as pd
import datetime as dt
import numpy as np
from tradepal.models.lstm import lstm

if __name__ == "__main__":
    symbols = ["SPY","DIA","QQQ","TLT","IWM"]
    lookback=[5, 10]
    n_units = [16,32,64]
    batch_size = [500,1000,2000]
    ln_rate=[0.001, 0.01, 0.1]
    
    params = pd.DataFrame(columns=['symbol',
                                   'lookback',
                                   'n_units',
                                   'batch_size',#4500
                                   'ln_rate',
                                   'train_loss',
                                  'val_loss',
                                  'train_acc',
                                  'val_acc',   
                                  'test_acc'],data=None)
   
    for symbol in symbols:
        print(dt.datetime.now())
        log = pd.DataFrame(columns=['lookback',
                                    'n_units',
                                   'batch_size',#4500
                                   'ln_rate',
                                   'train_loss',
                                  'val_loss',
                                  'train_acc',
                                  'val_acc',   
                                  'test_acc'],data=None)
        #different index funds have different start date
        if symbol=='SPY':
            sd=dt.datetime(1993,1,29)
        elif symbol=='DIA':
            sd=dt.datetime(1998,1,20)
        elif symbol=='QQQ':
            sd=dt.datetime(1999,3,10)
        elif symbol=='TLT':
            sd=dt.datetime(2002,7,30)
        elif symbol=='IWM':
            sd=dt.datetime(2000,5,26)
            
        count=0
        start_time=dt.datetime.now()
        for p in np.arange(len(lookback)):
            for i in np.arange(len(n_units)):
                for j in np.arange(len(batch_size)):
                    for k in np.arange(len(ln_rate)):
                        lstm_params = {'lookback': lookback[p],
                               'n_units':n_units[i],#64
                               'batch_size': batch_size[j],#500
                               'ln_rate' : ln_rate[k]} #0.005
         
                        #run the model,iterate through the 5 index funds "SPY","DIA","QQQ","TLT","IWM"
                        model, model_loss = lstm(lstm_params,symbol = symbol, sd=sd, 
                                          ed=dt.datetime(2020,8,31), impact=0.0)  
                        log = log.append({  
                                    'lookback'           : lookback[p],
                                    'n_units'            : n_units[i],
                                    'batch_size'         : batch_size[j],
                                    'ln_rate'            : ln_rate[k],
                                    'train_loss'         : model_loss['train_loss'],
                                    'val_loss'           : model_loss['val_loss'],
                                    'train_acc'          : model_loss['train_acc'],
                                    'val_acc'            : model_loss['val_acc'],
                                    'test_acc'           : model_loss['test_acc']}, ignore_index=True)
                        count +=1
                        if count%10 ==0:
                            print('Finished running the '+ str(count)+ 'th search for index fund '
                                  +symbol+', keep going!')
                            print(dt.datetime.now())
             
        end_time=dt.datetime.now()
        use_time=end_time - start_time
        use_minutes=(use_time.seconds+use_time.microseconds/1000000)/60.0  
        print('time used for grid searching for:  '+symbol+' is: '+str(use_minutes)+ ' minutes!')
        log.to_csv('tradepal/models/lstm_grid_search_'+symbol+'.csv', sep=',')
    
        #save the best hyperparameters
        test_acc=log['test_acc'].values
        max_acc_idx=np.argmax(test_acc)

        params = params.append({'symbol'         : symbol,  
                            'lookback'           : log.loc[max_acc_idx,'lookback'],
                            'n_units'            : log.loc[max_acc_idx,'n_units'],
                            'batch_size'         : log.loc[max_acc_idx,'batch_size'],
                            'ln_rate'            : log.loc[max_acc_idx,'ln_rate'],
                            'train_loss'         : log.loc[max_acc_idx,'train_loss'][-1],
                            'val_loss'           : log.loc[max_acc_idx,'val_loss'][-1],
                            'train_acc'          : log.loc[max_acc_idx,'train_acc'][-1],
                            'val_acc'            : log.loc[max_acc_idx,'val_acc'][-1],
                            'test_acc'           : log.loc[max_acc_idx,'test_acc']}, ignore_index=True)
        print('Finished seeking best hyperparameters for '+symbol+ '!')
    params.to_csv('tradepal/models/lstm_best_para.csv', sep=',')
    print('Best hyperparameters for all index funds are saved to csv file!')
             