#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:57:10 2020

@author: yalingliu
"""	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import os  		   	  	
import numpy as np		  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	
import datetime as dt 
from math import log 			  	 		  	
		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def symbol_to_path(symbol, base_dir=None):  		   	  			  	 		  		  		    	 		 		   		 		  
    """Return CSV file path given ticker symbol."""  		   	  			  	 		  		  		    	 		 		   		 		  
    if base_dir is None:  		   	  			  	 		  		  		    	 		 		   		 		  
        base_dir = os.environ.get("MARKET_DATA_DIR", './tradepal/data/')  		   	  			  	 		  		  		    	 		 		   		 		  
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def get_data(symbols, dates, addSPY=True, colname = 'Adj Close'):  		   	  			  	 		  		  		    	 		 		   		 		  
    """Read stock data (adjusted close) for given symbols from CSV files."""  		   	  			  	 		  		  		    	 		 		   		 		  
    df = pd.DataFrame(index=dates)  		   	  			  	 		  		  		    	 		 		   		 		  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    for symbol in symbols:  		   	  			  	 		  		  		    	 		 		   		 		  
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',  		   	  			  	 		  		  		    	 		 		   		 		  
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])  		   	  			  	 		  		  		    	 		 		   		 		  
        df_temp = df_temp.rename(columns={colname: symbol})  		   	  			  	 		  		  		    	 		 		   		 		  
        df = df.join(df_temp)  		   	  			  	 		  		  		    	 		 		   		 		  
        if symbol == 'SPY':  # drop dates SPY did not trade  		   	  			  	 		  		  		    	 		 		   		 		  
            df = df.dropna(subset=["SPY"])  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    return df 

def get_vol(symbols, dates, addSPY=True, colname = 'Volume'):  		   	  			  	 		  		  		    	 		 		   		 		  
    """Read stock data (adjusted close) for given symbols from CSV files."""  		   	  			  	 		  		  		    	 		 		   		 		  
    df = pd.DataFrame(index=dates)  		   	  			  	 		  		  		    	 		 		   		 		  	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    for symbol in symbols:  		   	  			  	 		  		  		    	 		 		   		 		  
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',  		   	  			  	 		  		  		    	 		 		   		 		  
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])  		   	  			  	 		  		  		    	 		 		   		 		  
        df_temp = df_temp.rename(columns={colname: symbol})  		   	  			  	 		  		  		    	 		 		   		 		  
        df = df.join(df_temp)  		   	  			  	 		  		  		    	 		 		   		 		  
        if symbol == 'SPY':  # drop dates SPY did not trade  		   	  			  	 		  		  		    	 		 		   		 		  
            df = df.dropna(subset=["SPY"])  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    return df 



def split_sequence(sequence, lookback):
    ''' split a univariate sequence into samples '''
    from numpy import array
    X, y = list(), list()
    for i in range(len(sequence)):
# find the end of this pattern
        end_ix = i + lookback
# check if we are beyond the sequence
       
        if end_ix > len(sequence)-1:
            break
# gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def genearte_input_sequence(sequence, lookback):
    ''' split a univariate sequence into samples '''
    from numpy import array
    X = list()
    for i in range(len(sequence)):
# find the end of this pattern
        end_ix = i + lookback
# check if we are beyond the sequence
       
        if end_ix > len(sequence)-1:
            break
# gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)        
    return array(X)

#calculate r2
def r2(y_true, y_pred):
    '''coefficient of determination'''
    from tensorflow.python.keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# calculate aic for regression
def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params
    return aic
    	 		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):  		   	  			  	 		  		  		    	 		 		   		 		  
    import matplotlib.pyplot as plt  		   	  			  	 		  		  		    	 		 		   		 		  
    """Plot stock prices with a custom title and meaningful axis labels."""  		   	  			  	 		  		  		    	 		 		   		 		  
    ax = df.plot(title=title, fontsize=12)  		   	  			  	 		  		  		    	 		 		   		 		  
    ax.set_xlabel(xlabel)  		   	  			  	 		  		  		    	 		 		   		 		  
    ax.set_ylabel(ylabel)  		   	  			  	 		  		  		    	 		 		   		 		  
    plt.show()  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def get_orders_data_file(basefilename):  		   	  			  	 		  		  		    	 		 		   		 		  
    return open(os.path.join(os.environ.get("ORDERS_DATA_DIR",'orders/'),basefilename))  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def get_learner_data_file(basefilename):  		   	  			  	 		  		  		    	 		 		   		 		  
    return open(os.path.join(os.environ.get("LEARNER_DATA_DIR",'Data/'),basefilename),'r')  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def get_robot_world_file(basefilename):  		   	  			  	 		  		  		    	 		 		   		 		  
    return open(os.path.join(os.environ.get("ROBOT_WORLDS_DIR",'testworlds/'),basefilename))  		   	  			  	 		  		  		    	 		 		   		 		  
