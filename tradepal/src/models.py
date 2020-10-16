#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:12:16 2020

@author: yalingliu
"""

import pandas as pd
import numpy as np
import datetime as dt 
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SVMSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import glob
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Dropout, Masking,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tradepal.src.util import split_sequence
from tradepal.src.indicators import get_XY_data	  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class models(object):	
    # constructor 	  		  		    	 		 		   		 		  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, symbol = "SPY", sd=dt.datetime(1993,1,29), ed=dt.datetime(2020,8,31), impact=0.0):  
        self.symbol = symbol
        self.impact = impact 
        self.sd = sd
        self.ed = ed
        
        df_dataX, df_dataY, df_indicators=get_XY_data(symbol = symbol, sd=dt.datetime(1993,1,29), 
                                                      ed=dt.datetime(2020,8,31), impact=0.0, recent_flag=False)
        
        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        df_dataY['encode']=labelencoder.fit_transform(df_dataY['dataY'])        
        
        X = df_dataX.values
        y=df_dataY['encode'].values 
        
        # split dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        
         ####Random Over-Sampling to combat imbalanced classes, with random replicas
        idx_hold=np.where(y_train==1)[0]
        idx_buy=np.where(y_train==0)[0]
        idx_sell=np.where(y_train==2)[0]              
        
        #get features and labels for each class
        sell_features=X_train[idx_sell]
        sell_labels=y_train[idx_sell]
        buy_features=X_train[idx_buy]        
        buy_labels=y_train[idx_buy]
        hold_features=X_train[idx_hold]
        hold_labels=y_train[idx_hold]
        
        #get the random index for replicas
        ids_buy = np.arange(len(idx_buy))
        ids_sell = np.arange(len(idx_sell))
        choices_buy = np.random.choice(ids_buy, len(idx_hold))
        choices_sell = np.random.choice(ids_sell, len(idx_hold))
        
        #set the replicas
        res_sell_features = sell_features[choices_sell]
        res_sell_labels = sell_labels[choices_sell]
        res_buy_features = buy_features[choices_buy]
        res_buy_labels = buy_labels[choices_buy]        
        
        ##combine the replicas of minority class with the majority class
        resampled_features = np.concatenate([res_sell_features, res_buy_features, hold_features], axis=0)
        resampled_labels = np.concatenate([res_sell_labels, res_buy_labels,hold_labels], axis=0)        
        
        #re-assign training data
        X_train=resampled_features
        y_train=resampled_labels        
        
        #shuffle the order
        order = np.arange(len(X_train))
        np.random.shuffle(order)
        X_train = X_train[order]
        y_train = y_train[order]      
      
        # pre-process training data, fit the scaler on training data only,
        # then standardise both training and test sets with the scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train=y_train
        self.y_test=y_test	 
        
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'Yaling Liu'  
    
    
    #---------------- 0: XGBoost ---------------------        
    def XGBoost(self):        
        base = XGBClassifier(n_estimators=200)
        max_depth_range=np.arange(10) + 1
        min_child_weight_range=np.arange(0.1,1,0.2)
        learning_rate_range=np.arange(0.001,0.1,0.005)
        params = {'max_depth': max_depth_range, 'min_child_weight':min_child_weight_range,
                  'learning_rate':learning_rate_range}
        model = RandomizedSearchCV(base ,param_distributions = params, cv=5, 
                                 scoring="balanced_accuracy", n_jobs= 10, verbose = 1)
        model.fit(self.X_train, self.y_train)
        
         # save the model to disk
        filename = 'tradepal/models/XGBoost_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        # make predictions for test data
        y_pred = model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        

    #---------------- 1: logistic regression ---------------------
    # all parameters not specified are set to their defaults
    def logReg(self):
        model = LogisticRegression(max_iter=1000)
        t_start = time.time()
        model.fit(self.X_train, self.y_train)
        
        t_end = time.time()
        train_time = t_end - t_start
        print('Training time: %f seconds' % train_time)
        
        # save the model to disk
        filename = 'tradepal/models/LogisticRegression_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))         
       
        t_start = time.time()
        y_pred = model.predict(self.X_test)
        t_end = time.time()
        predict_time = t_end - t_start
        print('Prediction time: %f seconds' % predict_time)        
        
        #calculate accuracy
        correct_idx=np.where(self.y_test-y_pred==0)[0]
        accuracy=correct_idx.shape[0]/y_pred.shape[0]
        print('prediction accuracy is: %f' % accuracy)
        
        return accuracy,train_time


    # ---------------- 2: Random Forest ---------------------        
    def randForest(self):
        base = RandomForestClassifier(n_estimators=200)  # 200 is a large but fair number
        max_depth_range = np.arange(10) + 1
        tuned_params = {'max_depth': max_depth_range}
        model = RandomizedSearchCV(base, param_distributions=tuned_params, scoring='balanced_accuracy', cv=5, iid=False)
        
        t_start = time.time()
        model.fit(self.X_train, self.y_train)
        t_end = time.time()
        train_time = t_end - t_start
        print('Training time: %f seconds' % train_time)
        
        # save the model to disk      
        filename = 'tradepal/models/RandomForestClassifier_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))   
        
        # find best fit parameters
        best_dt_parameter = model.best_params_
        print("Best max_depth parameter for random forest: {}".format(best_dt_parameter))
        
        t_start = time.time()
        y_pred = model.predict(self.X_test)
        t_end = time.time()
        predict_time = t_end - t_start
        print('Prediction time: %f seconds' % predict_time)    
        
        #calculate accuracy
        correct_idx=np.where(self.y_test-y_pred==0)[0]
        accuracy=correct_idx.shape[0]/y_pred.shape[0]
        print('prediction accuracy is: %f' % accuracy)         
        return accuracy,train_time


    # ---------------------------- 3: AdaBoosting -----------------------------
    def adaBst(self):
        param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"]
              }        
        DTC = DecisionTreeClassifier(random_state = 1, max_features = "auto", class_weight = "balanced",max_depth=10)        
        base = AdaBoostClassifier(base_estimator = DTC)        
        # run grid search
        model = GridSearchCV(base, param_grid=param_grid, scoring = 'f1_weighted')
        
        
        # train
        t_start = time.time()
        model = model.fit(self.X_train, self.y_train)
        t_end = time.time()
        train_time = t_end - t_start
        print('Training time: %f seconds' % train_time)        
        
        filename = 'tradepal/models/AdaBoostClassifier_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        # predict
        t_start = time.time()
        y_pred = model.predict(self.X_test)
        t_end = time.time()
        predict_time = t_end - t_start
        print('Prediction time: %f seconds' % predict_time)
        
        #calculate accuracy
        correct_idx=np.where(self.y_test-y_pred==0)[0]
        accuracy=correct_idx.shape[0]/y_pred.shape[0]
        print('prediction accuracy is: %f' % accuracy)
        return accuracy,train_time  	  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  


    # ---------------------- 4: Support Vector Machine -----------------------
    def SVM_model(self):
         # hyperparameter tuning
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]        
        param_grid = {'C': Cs, 'gamma' : gammas}
        model = RandomizedSearchCV(svm.SVC(kernel='rbf'), 
                                   param_grid, scoring = 'balanced_accuracy',cv=5)       
        
        t_start = time.time()
        model.fit(self.X_train, self.y_train)
        t_end = time.time()
        train_time = t_end - t_start
        print('Training time: %f seconds' % train_time)        
        
        filename = 'tradepal/models/SVM_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        # find best fit parameters
        best_parameter = model.best_params_
        print("Best size parameter for SVM: {}".format(best_parameter))
        
        # predict
        t_start = time.time()
        y_pred = model.predict(self.X_test)
        t_end = time.time()
        predict_time = t_end - t_start
        print('Prediction time: %f seconds' % predict_time)
        
        #calculate accuracy
        correct_idx=np.where(self.y_test-y_pred==0)[0]
        accuracy=correct_idx.shape[0]/y_pred.shape[0]
        print('prediction accuracy is: %f' % accuracy)
        return accuracy,train_time       
    
        
        
    # ---------------------- 5: long short-term memory (lstm) -----------------------
    def lstm(self, lstm_params):
        #NOTE: lstm does not use any data augmentation techniques because the data need to 
        #preserve their original temporal sequence to track system memory
        df_dataX, df_dataY, df_indicators=get_XY_data(self.symbol, sd=self.sd, ed=self.ed, 
                                                      impact=self.impact,recent_flag=False)
            
        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        df_dataY['encode']=labelencoder.fit_transform(df_dataY['dataY'])        
           
        X = df_dataX.values
        y=df_dataY['encode'].values        
        
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
        
        
        # define model, good to go    
        model = Sequential()
        model.add(Masking(mask_value=-1, input_shape=(lstm_params['lookback'], X_tr.shape[2])))
        model.add(LSTM(lstm_params['n_units'], activity_regularizer = l2(0.001)))    
        model.add(LeakyReLU(alpha=0.1))
        #output layer
        model.add(Dense(y_tr.shape[1], activation='softmax')) 
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lstm_params['ln_rate']), metrics=['acc'])
       
        #fit the model
        history = model.fit(X_tr, y_tr,
                            batch_size= lstm_params['batch_size'], epochs=15, verbose=0,  
                            validation_split=0.1)    
       
        # split test set into samples
        X_ts, y_ts = split_sequence(test_ds, lstm_params['lookback'])
        y_ts = y_ts[:,-1:]
        X_ts=X_ts[:,:,:-1]       
        
        #evaluate accuray
        score = model.evaluate(X_ts, y_ts, verbose=0)
        print("Test Accuracy:", score[1])
        
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


if __name__=="__main__":  	
    symbols = ["SPY","DIA","QQQ","TLT","IWM"]
    mods = ['LogisticRegression','RandomForestClassifier','AdaBoostClassifier','SVM']
    #mod_accuracy save accuracy for each fund and each ML method, similar for train_time
    mod_accuracy = pd.DataFrame(columns=mods,data=None,index=symbols)
    train_time = pd.DataFrame(columns=mods,data=None,index=symbols)
    for symbol in symbols:
    #datetime(1993,1,29) is the earliest start date for the 5 funds, it does not matter if 
    #a fund's start date is later, because missing dates will be omited  
        ML=models(symbol = symbol, sd=dt.datetime(1993,1,29), ed=dt.datetime(2020,8,31), impact=0.0) 
        mod_accuracy.loc[symbol,'LogisticRegression'],train_time.loc[symbol,'LogisticRegression']=ML.logReg()
        mod_accuracy.loc[symbol,'RandomForestClassifier'],train_time.loc[symbol,'RandomForestClassifier']=ML.randForest()
        mod_accuracy.loc[symbol,'AdaBoostClassifier'],train_time.loc[symbol,'AdaBoostClassifier']=ML.adaBst()
        mod_accuracy.loc[symbol,'SVM'],train_time.loc[symbol,'SVM']=ML.SVM_model()
        print('Finished saving model for '+ symbol+' at '+str(dt.datetime.now()))
        
    mod_accuracy.to_csv('tradepal/models/LR_RF_AB_SVM_accuracy.csv', sep=',')
    train_time.to_csv('tradepal/models/LR_RF_AB_SVM_train_time.csv', sep=',')


    #train and save lstm models to disk
    filename=glob.glob('tradepal/models/lstm_best_para.csv')
    best_para=pd.read_csv(filename[0], sep=',', delimiter=None, header='infer') 
    best_para=best_para.set_index('symbol')
    
    for symbol in symbols:        
        lstm_params = {
                    'lookback':best_para.loc[symbol,'lookback'].astype(int),
                   'n_units':best_para.loc[symbol,'n_units'].astype(int),
                   'batch_size': best_para.loc[symbol,'batch_size'].astype(int),
                   'ln_rate' : best_para.loc[symbol,'ln_rate']} 
        
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
     
        #run the model
        start_time=dt.datetime.now()
        model, model_loss = ML.lstm(lstm_params)         
        print("Test accuracy for "+ symbol +" is "+ str(model_loss['test_acc']))
        
        # serialize model to JSON
        name='tradepal/models/lstm_'+symbol
        model_json = model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name+ ".h5")
        print("Saved lstm model to disk for "+symbol+"!")
        
        end_time=dt.datetime.now()
        use_time=end_time - start_time
        use_minutes=(use_time.seconds+use_time.microseconds/1000000)/60.0  
        print('time used: '+str(use_minutes)+ ' minutes!')