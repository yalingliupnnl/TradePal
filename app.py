#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:57:10 2020

@author: yalingliu
"""

#!/usr/bin/env python
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import datetime as dt
import pandas as pd
import numpy as np
# import ManualStrategy as ms 
#import StrategyLearner as sl
#from marketsimcode import compute_portvals,compute_port_stats
import matplotlib.pyplot as plt
#import seaborn as sns
#import calmap        
#from indicators import normalize_prices
from tradepal.models.recommend import recommend #recommend_today
from tradepal.indicators import get_XY_data
import yfinance as yf
from datetime import timedelta


# Start execution
def main():
    # Determine app mode to run
    # Render mardown text    
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Welcome page", "Run the app", "Data Exploration"])
    if app_mode == "Welcome page":
        welcome()
    elif app_mode == "Run the app":
        run_app()
    elif app_mode == "Data Exploration":
        run_data_analysis()
    # elif app_mode == "Model deep dive":
    #     run_model_analysis()

def welcome():
    f = open("tradepal/resources/intro.md", 'r')
    st.markdown(f.read())
    f.close()
    st.image(load_image('tradepal/resources/charging-bull.jpg'), use_column_width=True)
    "Charging Bull at the Wall Street"

# Main app
def run_app():   
    # Filters
    # Select model to use
    
    st.title("Trading Helper")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Trading Helper App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)    
    np.random.seed(12345678)
    fund_name= st.sidebar.selectbox("Select Index Fund",("SPY (SP 500)","DIA (Dow Jones Industrial Average)",
                                                         "QQQ (Nasdaq 100)","TLT (U.S. Treasury 20+ Year Bond)",  
                                                         "IWM (Russell 2000)"))
    
    model_name= st.sidebar.selectbox("Select Model",("Logistic Regression",
                                                         "Random Forest","Ada Boosting", 
                                                         "Support Vector Machine",
                                                         "Long Short-Term Memory"))
   
    if fund_name=="SPY (SP 500)":
        symbol='SPY'
    elif fund_name=="DIA (Dow Jones Industrial Average)":
        symbol='DIA'
    elif fund_name=="QQQ (Nasdaq 100)":
        symbol='QQQ'
    elif fund_name=="TLT (U.S. Treasury 20+ Year Bond)":
        symbol='TLT'
    elif fund_name=="IWM (Russell 2000)":
        symbol='IWM'
        
    if model_name=="Logistic Regression":
        mod='LogisticRegression'
    elif model_name=="Random Forest":
        mod='RandomForestClassifier'
    elif model_name=="Ada Boosting":
        mod='AdaBoostClassifier'
    elif model_name=="Support Vector Machine":
        mod='SVM'
    elif model_name=="Deep Neural Network":
        mod='DNN'
    elif model_name=="Long Short-Term Memory":
        mod='lstm'

    y_option, model =recommend(symbol,mod)   
 
    
    result=""
    if y_option==0:
        result='BUY'
    elif y_option==2:
        result='SELL'
    else:
        result='HOLD'
        
    "   "
    "   "    

    if st.button("Predict"):
        st.success(f'### The predicted trading for today is:  {result}')

    "   "
    "   "

    final,adaModel=recommend(symbol,'AdaBoostClassifier') 
    if final==0:
        final='BUY'
    elif final==2:
        final='SELL'
    else:
        final='HOLD'
    
    if st.sidebar.checkbox('Show recommendation'):
        st.success(f'### The recommended trading for today is:  {final}')
        
        
    if st.sidebar.checkbox('Show backtesting results of all models'):
        st.write("### Backtesting results for the past 3 years is shown below (each year has 252 transaction days), where green color and red color represent right and wrong predictions, respectively.")
        image_name='tradepal/resources/back_test_model_performance_'+symbol+'.png'
        st.image(load_image(image_name), use_column_width=True)
        "   "
        "   "
        st.write("### Backtesting accuracy of all models for index fund "+ symbol+" in the past 3 years is shown below:")
        ##show picture of model performance here
        image_name='tradepal/resources/back_test_performance_barchart_'+symbol+'.png'
        st.image(load_image(image_name), use_column_width=True)
    
# Data Analysis app
def run_data_analysis(): 
    
    st.title('Index Fund Data Analysis')
    
    st.write("### Historical normalized prices variations for the 5 representative index funds is shown below:")
    st.image(load_image('tradepal/resources/price_variations.png'), use_column_width=True)
    
    "   "
    st.write('### Below we explore the SPY fund index stock data in 2011.')
    
    symbol='SPY'			
    df_dataX, df_dataY, indicators=get_XY_data(symbol, sd=dt.datetime(2010,1,1), ed=dt.datetime(2010, 12, 31),impact=0.0)   	  			  	 		  		  		    	 		 		   		 		  
   
    
    # Dataframe samples
    st.subheader("Sample of raw dataset")
    raw=pd.read_csv("tradepal/data/SPY.csv")     
    st.write(raw.iloc[176:185,])
    st.subheader("Features used in training")
    st.dataframe(df_dataX[30:35])
    st.write("Note: SMA=simple moving average, BB_pct= Bollinger Band percent, CCI=Commodity Channel Index, OBV=On-Balance Volume, macd=moving average convergence divergence, rsi_12=Relative strength index (window is 12), kdjj=stochastic oscillator, adx=Average directional index   ")

    st.subheader("Trading performance of different methods measured by normalized market value")
    
    "During model training period for the SPY fund index, the performance of random forest is much better than those of manual strategy (based on certain trading rules) and benchmark (no trading)"
    st.image(load_image('tradepal/resources/SPY_training.png'), use_column_width=True)
    
    "Similarly, during model testing period for the SPY fund index, the performance of random forest is also much better than manual strategy and benchmark."
  
    st.image(load_image('tradepal/resources/SPY_testing.png'), use_column_width=True)



@st.cache
def load_image(path):
	im =Image.open(path)
	return im   
    

if __name__=='__main__':
    main()
