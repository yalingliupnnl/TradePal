#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:57:10 2020

@author: yalingliu
"""
 			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		
import pandas as pd   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt 
from datetime import timedelta	  		  		    	 		 		   		 		  
from tradepal.util import get_data, get_vol,plot_data  
import matplotlib.pyplot as plt		  
import stockstats
import copy
import yfinance as yf
import logging


def author():  		   	  			  	 		  		  		    	 		 		   		 		  
    return 'Yaling Liu'  

def OBV(symbol, sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),recent_flag=False):
    pd.options.mode.chained_assignment = None    		   	  			  	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  
    syms=[symbol]		   	  		  	 		  		  		    	 		 		   		 		  
    
    if not recent_flag:
        prices=get_data(syms, dates) 
        vol = get_vol(syms, dates)
    else:
        yfdata = yf.download(symbol, start=dt.datetime.now().date()-dt.timedelta(1115), 
                             end=dt.datetime.now().date(), progress=False)
        prices=pd.DataFrame(index=yfdata.index,columns=[symbol],data=yfdata['Adj Close'].values) 
        vol=pd.DataFrame(index=yfdata.index,columns=[symbol],data=yfdata['Volume'].values) 
    OBV_df = pd.DataFrame(index=prices.index,columns=['OBV'],data=0)  	

    close_diff=prices[1:].values-prices[:-1]#len(df.index)-1 
    i=1
    for index, data in close_diff.iterrows():#close_diff starts from the 2nd day of dates       
        if close_diff.loc[index,symbol]>0:
            OBV_df['OBV'].iloc[i]=OBV_df['OBV'].iloc[i-1]+vol.iloc[i,0]
        elif close_diff.loc[index,symbol]<0:
            OBV_df['OBV'].iloc[i]=OBV_df['OBV'].iloc[i-1]-vol.iloc[i,0]
        else:
            OBV_df['OBV'].iloc[i]=OBV_df['OBV'].iloc[i-1]
        i +=1
    return OBV_df
	 	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def compute_indicators(symbol, sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),recent_flag=False):
    pd.options.mode.chained_assignment = None  # default='warn'    
    if not recent_flag:
        stock = stockstats.StockDataFrame.retype(pd.read_csv('tradepal/data/'+symbol+'.csv')) 
        OBV_df=OBV(symbol, sd,ed)
        dates = pd.date_range(sd, ed)		
        Adj_close_price=get_data([symbol],dates)	 
       
    else:
        yfdata = yf.download(symbol, start=dt.datetime.now().date()-dt.timedelta(1115), 
                             end=dt.datetime.now().date(), progress=False)        
        Adj_close_price=pd.DataFrame(index=yfdata.index,columns=[symbol],data=yfdata['Adj Close'].values) 
         
        yfdata.reset_index(inplace=True)
        stock = stockstats.StockDataFrame.retype(yfdata) 
        OBV_df=OBV(symbol, sd,ed,recent_flag=True)
    

    stock2=stock[['close_3_trix','macd','rsi_12','rsi_6','kdjj','adx','cr-ma1','cr-ma2']]	   	  			  	 		  		  		    	 		 		   		 		  
#    stock['macd']	#Moving average convergence divergence 	   	
#    stock['rsi_12'] #Relative strength index 
#    stock['kdjj']#Stochastic oscillator		  	
#    stock['adx']#Average directional index (ADX)
#    stock['std']#MSTD: moving standard deviation	
    
     #normalize the prices of sybols and SPY
    prices_norm=normalize_prices(Adj_close_price[symbol])
    
    #create a dataframe to save the 3 indicators
    df_indicators=pd.DataFrame(index=prices_norm.index,data=None)
    
    #simple moving average (SMA)
    df_indicators['price']=prices_norm
    df_indicators['SMA']=prices_norm.rolling(window=14, center=False).mean()
    
    #Bollinger Band percent(BB_pct)
    std=prices_norm.rolling(window=14,center=False).std()
    df_indicators['upper']=df_indicators['SMA']+2*std
    df_indicators['lower']=df_indicators['SMA']-2*std
    df_indicators['BB_pct']=(prices_norm-df_indicators['lower'])/(df_indicators['upper']-df_indicators['lower'])
    
    #commodity channel index (CCI)
    df_indicators['CCI']=(prices_norm-df_indicators['SMA'])/(1.5*prices_norm.std())    
    df_indicators['OBV']= OBV_df['OBV']
    
    df_indicators = df_indicators.join(stock2)

    return df_indicators	

def get_XY_data(symbol = "SPY", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), impact=0.0,recent_flag=False): 
    pd.options.mode.chained_assignment = None  # default='warn'    
    logging.disable(logging.CRITICAL)
    if not recent_flag:        
        dates = pd.date_range(sd, ed)		
        prices=get_data([symbol],dates)	 
        df_indicators=compute_indicators(symbol,sd,ed,recent_flag)
    else:
        start=dt.datetime.now().date()-dt.timedelta(1115)
        end=dt.datetime.now().date()
        yfdata = yf.download(symbol, start=dt.datetime.now().date()-dt.timedelta(1115), 
                             end=dt.datetime.now().date(), progress=False)     
        prices=pd.DataFrame(index=yfdata.index,columns=[symbol],data=yfdata['Adj Close'].values) 
        df_indicators=compute_indicators(symbol,start,end,recent_flag=True)	   	  			  	 		  		  		    	 		 		   		 		   		   	  			  	 		  		  		    	 		 		   		 		  
  		   
    #construct the input data dataX for training    
    indicators=df_indicators[['SMA', 'BB_pct', 'CCI', 'OBV', 
                              'close_3_trix','macd','rsi_12','rsi_6','kdjj','adx','cr-ma1','cr-ma2']]
    indicators.fillna(0,inplace=True)    
    
    #use deepcopy
    indicators_copy = copy.deepcopy(indicators)
    df_dataX=indicators_copy[13:-2] #e.g., during a calendar year 1/1-12/31, the df_dataX span from 1/14-12/29, 
    #from the 14th day to the last 3th day, omit the first 13 days and the last 2 days
    #because for the 1st 13 days there are no SMA, BB_pct and CCI values due to window size of 14
    #ends on 12/29, because condictions on 12/29 predict the actual trading on 12/30, which is based on price on 12/30-12/31
    
    #construct the input data dataY for training
    dataY=[]
    YBUY=0.01+impact
    YSELL=-0.01-impact
    # window=2, use indicators of a certain day to predict trading of the next day
    #the trading label on 12/29 (whether to buy, sell or hold next day) is for actual trading on 12/30 
    for i in np.arange(len(df_dataX.index))+14:
        #ideal trading option from 1/15-12/30 (shift 1 day after that of df_dataX ),
        #because the features on 14th is used to predict optimal actual trading on the 15th.        
        
        ret=(prices.iloc[i+1]/prices.iloc[i])-1 #2-day return
        ret=ret.values
        #buy
        if ret>YBUY:
            dataY.append("BUY")#0
        #sell
        elif ret<YSELL:
            dataY.append("SELL")#2
        else:
            dataY.append("HOLD")#1
    
    dataY=np.array(dataY)	      
    df_dataY=pd.DataFrame(index=df_dataX.index, columns=["dataY"],data=dataY) 
    
    #add libor interest rate and treasury bond rate
    libor=pd.read_csv('tradepal/data/historical-libor-rates.csv',sep=',', index_col=0,
                  delimiter=None, header='infer')	
    libor=libor.drop(['3M','6M','12M'],axis=1)
    treasury=pd.read_csv('tradepal/data/30-year-treasury-bond-rate.csv',sep=',', index_col=0,
                      delimiter=None, header='infer')	
    #join with df_dataX
    libor1=df_dataY.join(libor)  		
    fill_missing(libor1)	
    fill_missing(treasury)	
    libor1=libor1.join(treasury)	
    libor1=libor1.drop(['dataY'],axis=1) 		
    df_dataX=df_dataX.join(libor1)
    fill_missing(df_dataX)
    
    #join with indicators
    libor2=indicators.join(libor)  		
    fill_missing(libor2)		
		
    indicators=libor2.join(treasury)
    fill_missing(indicators)    
  
    return df_dataX, df_dataY, indicators
    #the prediction of trading on 10/1/2020 compares with actual optimal trading on 10/2/2020	
  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def normalize_prices(prices):  		
    fill_missing(prices)   
    return prices/prices.iloc[0]	  

def fill_missing(prices)	:
    #fill forward first, and then fill backward
    prices.fillna(method='ffill',inplace=True)	
    prices.fillna(method='bfill',inplace=True)		

def plot_price():
    symbols=["SPY","DIA","QQQ","TLT","IWM"]  
    sd=dt.datetime(1993,1,29)
    ed=dt.datetime.now()
    dates = pd.date_range(sd, ed)
    prices=get_data(['SPY'],dates)
    for symbol in symbols:		
        temp_price=get_data([symbol],dates)
        prices=prices.join(temp_price)
    
    prices=normalize_prices(prices)

    prices=prices.rename(columns={"SPY":"SPY (SP 500)","DIA":"DIA (Dow Jones Industrial Average)",
                       "QQQ":"QQQ (Nasdaq 100)","TLT":"TLT (U.S. Treasury 20+ Year Bond)",
                       "IWM":"IWM (Russell 2000)"})
    
    ax=prices.plot(title='stock price variations',figsize=(20,10), fontsize=20)	  		    	 		 		   		 		  
    ax.set_xlabel('Date', fontsize=20)  		   	  			  	 		  		  		    	 		 		   		 		  
    ax.set_ylabel('Price (normalized)', fontsize=20)  
    ax.legend(loc='best', fontsize=20) 
    
    plt.grid(which='both')
    ax.set_xlim([sd, ed])
    plt.rcParams.update({'font.size': 20})    
    plt.tight_layout()
    fig_name='tradepal/resources/price_variations.png'
    plt.savefig(fig_name)	
    plt.close()  	 		  		  		    	 		 		   		 		  

def test_code():
    start_date=dt.datetime(2008,1,1)
    end_date=dt.datetime(2009,12,31)
    symbol='SPY'
    
    #SMA
    ax=(compute_indicators(symbol,start_date,end_date)[['price','SMA']].plot
        (title='Simple Moving Average (SMA)',figsize=(20,10), fontsize=12))	  		    	 		 		   		 		  
    ax.set_xlabel('Date', fontsize=15)  		   	  			  	 		  		  		    	 		 		   		 		  
    ax.set_ylabel('Price (normalized)', fontsize=15)  
    ax.legend(loc='best') 
    
    plt.grid(which='both')
    ax.set_xlim([dt.datetime(2008,1,1), dt.datetime(2009,12,31)])
    plt.rcParams.update({'font.size': 15})    
    fig_name='SMA.png'
    plt.savefig(fig_name)	
    plt.close()
    
    #BB_pct
    ax=(compute_indicators(symbol,start_date,end_date)[['upper','SMA','lower','BB_pct']].plot
        (title='Bollinger Bands percent (%B)', figsize=(20,10), fontsize=12))	  		    	 		 		   		 		  
    ax.set_xlabel('Date')  		   	  			  	 		  		  		    	 		 		   		 		  
    ax.set_ylabel('Price (normalized)') 
    plt.axhline(y=0.2, linestyle='--')
    plt.axhline(y=0.8, linestyle='--')
    ax.legend(['upper','SMA','lower','%B'],loc='best') 
    plt.grid(which='both')
    ax.set_xlim([dt.datetime(2008,1,1), dt.datetime(2009,12,31)])
    plt.rcParams.update({'font.size': 15})   
    fig_name='BB_pct.png'
    plt.savefig(fig_name)
    plt.close()	
    
    #CCI
    ax=(compute_indicators(symbol,start_date,end_date)[['price','CCI']].plot
        (title='Commodity channel index (CCI)', figsize=(20,10), fontsize=12))	  		    	 		 		   		 		  
    ax.set_xlabel('Date')  		   	  			  	 		  		  		    	 		 		   		 		  
    ax.set_ylabel('Price (normalized)')  
    plt.axhline(y=0, linestyle=':')
    plt.axhline(y=-.5, linestyle='--')
    plt.axhline(y=.5, linestyle='--')
    ax.legend(loc='best') 
    plt.grid(which='both')
    ax.set_xlim([dt.datetime(2008,1,1), dt.datetime(2009,12,31)])
    plt.rcParams.update({'font.size': 15})   
    fig_name='CCI.png'
    plt.savefig(fig_name) 	
    plt.close()	  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    test_code()  		   	  			  	 		  		  		    	 		 		   		 		  
