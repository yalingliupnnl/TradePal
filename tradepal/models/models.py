#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:12:16 2020

@author: yalingliu
"""


import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor,AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV,RandomizedSearchCV
from sklearn import metrics, svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from tradepal.indicators import get_XY_data
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score
from imblearn.over_sampling import SVMSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedStratifiedKFold
#from imblearn.pipeline import Pipeline


 	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class models(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, symbol = "SPY", sd=dt.datetime(1993,1,29), ed=dt.datetime(2020,8,31), impact=0.0):  
        self.symbol = symbol
        self.impact = impact 
        
        df_dataX, df_dataY, df_indicators=get_XY_data(symbol = symbol, sd=dt.datetime(1993,1,29), 
                                                      ed=dt.datetime(2020,8,31), impact=0.0, recent_flag=False)
        
        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        df_dataY['encode']=labelencoder.fit_transform(df_dataY['dataY'])
        
        
        X = df_dataX.values
        # y = enc_df.values
        y=df_dataY['encode'].values
#        len(np.where(y==1)[0])/y.shape[0]#HOLD, 0.737 for SPY, 0.924 for TLT, 0.88 DIA
#        len(np.where(y==0)[0])/y.shape[0]#BUY, 0.138 for SPY,  0.037 for TLT,0.061 DIA
#        len(np.where(y==2)[0])/y.shape[0]#SELL, 0.125 for SPY, 0.038 for TLT, 0.579 DIA      
       
        
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
        
        res_sell_features.shape
        ##combine the replicas of minority class with the majority class
        resampled_features = np.concatenate([res_sell_features, res_buy_features, hold_features], axis=0)
        resampled_labels = np.concatenate([res_sell_labels, res_buy_labels,hold_labels], axis=0)
        
        resampled_features.shape
        #re-assign training data
        X_train=resampled_features
        y_train=resampled_labels
        # summarize class distribution
        counter = Counter(y_train)
        print(counter) 
        
        ####add random under-resampling
        # define undersample strategy, it seems not work for multi-class after over-sampling
        # because the sampling_strategy cannot be assigned a float number
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=1)#majority
        # fit and apply the transform
        X_train, y_train = undersample.fit_resample(X_train, y_train)
        
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
        
    def XGBoost(self):    
    # fit model no training data
        base = XGBClassifier(n_estimators=200)
        max_depth_range=np.arange(10) + 1
        min_child_weight_range=np.arange(0.1,1,0.2)
        learning_rate_range=np.arange(0.001,0.1,0.005)
        params = {'max_depth': max_depth_range, 'min_child_weight':min_child_weight_range,
                  'learning_rate':learning_rate_range}
        model = RandomizedSearchCV(base ,param_distributions = params, cv=5, 
                                 scoring="balanced_accuracy", n_jobs= 10, verbose = 1)
        model.fit(self.X_train, self.y_train)
#        model.fit(X_train, y_train)
        
         # save the model to disk
        filename = 'tradepal/models/XGBoost_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        # make predictions for test data
        y_pred = model.predict(self.X_test)
#        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.y_test, predictions)
#        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))#0.877 for random over-sampling
        #0.641 for SVMSMOTE, 0.69 for random over-sampling(direct replica),
        #0.6543 for SVMSMOTE, with both random over-sampling and under-sampling
        #0.648 for random over-sampling(random replica),
        #0.677 for random over-sampling(random replica),with under-sampling
        # 0.597 for regular SMOTE

    #---------------- 1: logistic regression ---------------------
    # all parameters not specified are set to their defaults
    def logReg(self):
        model = LogisticRegression()
        t_start = time.time()
        model.fit(self.X_train, self.y_train)
#        model.fit(X_train, y_train)
        t_end = time.time()
        train_time = t_end - t_start
        print('Training time: %f seconds' % train_time)
        
        # save the model to disk
        filename = 'tradepal/models/LogisticRegression_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))
         
        # load the model from disk
        # model = pickle.load(open(filename, 'rb'))
        
        t_start = time.time()
        y_pred = model.predict(self.X_test)
#        y_pred = model.predict(X_test)
        t_end = time.time()
        predict_time = t_end - t_start
        print('Prediction time: %f seconds' % predict_time)
        
        
        #calculate accuracy
        correct_idx=np.where(self.y_test-y_pred==0)[0]
#        correct_idx=np.where(y_test-y_pred==0)[0]
        accuracy=correct_idx.shape[0]/y_pred.shape[0]#0.57
        print('prediction accuracy is: %f' % accuracy)
        #0.62 for 1% 3-day return, 0.76 for 1.5% 3-day return, 0.735 for 1% 2-day return
        #0.4806 for 1% 2-day return with random over-sampling(direct replicas)
        #0.53 for 1% 2-day return with random over-sampling(random replicas)
        return accuracy,train_time


        # ---------------- 2: Random Forest ---------------------
        # base = RandomForestRegressor(n_estimators=200) # 200 is a large but fair number
    def randForest(self):
        base = RandomForestClassifier(n_estimators=200)
        # tune parameter: max tree depth 10
#        scoring = metrics.make_scorer(metrics.mean_squared_log_error, greater_is_better=False)
        max_depth_range = np.arange(10) + 1
        tuned_params = {'max_depth': max_depth_range}
        model = RandomizedSearchCV(base, param_distributions=tuned_params, scoring='balanced_accuracy', cv=5, iid=False)
        
        t_start = time.time()
        model.fit(self.X_train, self.y_train)
#        model.fit(X_train, y_train)
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
#        y_pred = model.predict(X_test)
        t_end = time.time()
        predict_time = t_end - t_start
        print('Prediction time: %f seconds' % predict_time)    
        
        #calculate accuracy
        correct_idx=np.where(self.y_test-y_pred==0)[0]
#        correct_idx=np.where(y_test-y_pred==0)[0]
        accuracy=correct_idx.shape[0]/y_pred.shape[0]#0.57
        print('prediction accuracy is: %f' % accuracy)
         #0.7505 for 1% 2-day return with random over-sampling
         #0.59 for random over-sampling(random replica),
        return accuracy,train_time


        # ---------------------- 3: Boosting -----------------------
    def adaBst(self):
#        base = DecisionTreeClassifier(max_depth=10)
        param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"]
              }        
        DTC = DecisionTreeClassifier(random_state = 1, max_features = "auto", class_weight = "balanced",max_depth=10)        
        base = AdaBoostClassifier(base_estimator = DTC)        
        # run grid search
        model = GridSearchCV(base, param_grid=param_grid, scoring = 'f1_weighted')#balanced_accuracy
        
#        model = AdaBoostClassifier(base_estimator=base, n_estimators=200, random_state=1)
        
        # train
        t_start = time.time()
        model = model.fit(self.X_train, self.y_train)
        #model = model.fit(X_train, y_train)
        t_end = time.time()
        train_time = t_end - t_start
        print('Training time: %f seconds' % train_time)        
        
        filename = 'tradepal/models/AdaBoostClassifier_' + self.symbol+ '.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        # predict
        t_start = time.time()
        y_pred = model.predict(self.X_test)
        #y_pred = model.predict(X_test)
        t_end = time.time()
        predict_time = t_end - t_start
        print('Prediction time: %f seconds' % predict_time)
        
        #calculate accuracy
        correct_idx=np.where(self.y_test-y_pred==0)[0]
#        correct_idx=np.where(y_test-y_pred==0)[0]
        accuracy=correct_idx.shape[0]/y_pred.shape[0]#0.57
        print('prediction accuracy is: %f' % accuracy)#0.76 for 1.5%
        #0.9417 for 1% 2-day return with random over-sampling
        #0.627 for SVMSMOTE, 
        #0.635 for SVMSMOTE, with both over- and under-resampling
        #0.704 for random over-sampling(direct replica)
        #0.71 for random over-sampling(random replica),
        #0.709 for random over-sampling(random replica),and under-sampling
        #0.581 for regular SMOTE
        return accuracy,train_time  #0.563		  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  


# ---------------------- 4: Support Vector Machine, Y need to be classifier -----------------------
    def SVM_model(self):
         # hyperparameter tuning
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]        
        param_grid = {'C': Cs, 'gamma' : gammas}
        model = RandomizedSearchCV(svm.SVC(kernel='rbf'), 
                                   param_grid, scoring = 'balanced_accuracy',cv=5)       
        # class_weight='balanced' make the test accuracy much worse 
        # training
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
        accuracy=correct_idx.shape[0]/y_pred.shape[0]#0.57
        print('prediction accuracy is: %f' % accuracy)
        return accuracy,train_time#0.58 for 1%, 0.76 for 1.5% 3-day return 
        #0.7245 for 1% 2-day return with random over-sampling

if __name__=="__main__":  	
    symbols = ["SPY","DIA","QQQ","TLT","IWM"]
    mods = ['LogisticRegression','RandomForestClassifier','AdaBoostClassifier','SVM']
    #mod_accuracy save accuracy for each fund and each ML method, similar for train_time
    mod_accuracy = pd.DataFrame(columns=mods,data=None,index=symbols)
    train_time = pd.DataFrame(columns=mods,data=None,index=symbols)
    for symbol in symbols:
    #datetime(1993,1,29) is the earliest start date for the 5 funds, it does not matter if 
    #a fund's start date is later, because missing dates will be omited  
        model=models(symbol = symbol, sd=dt.datetime(1993,1,29), ed=dt.datetime(2020,8,31), impact=0.0) 
        mod_accuracy.loc[symbol,'LogisticRegression'],train_time.loc[symbol,'LogisticRegression']=model.logReg()
        mod_accuracy.loc[symbol,'RandomForestClassifier'],train_time.loc[symbol,'RandomForestClassifier']=model.randForest()
        mod_accuracy.loc[symbol,'AdaBoostClassifier'],train_time.loc[symbol,'AdaBoostClassifier']=model.adaBst()
        mod_accuracy.loc[symbol,'SVM'],train_time.loc[symbol,'SVM']=model.SVM_model()
        print('Finished saving model for '+ symbol+' at '+str(dt.datetime.now()))
        
    mod_accuracy.to_csv('tradepal/models/LR_RF_AB_SVM_accuracy.csv', sep=',')
    train_time.to_csv('tradepal/models/LR_RF_AB_SVM_train_time.csv', sep=',')

