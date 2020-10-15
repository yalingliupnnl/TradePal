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
   
    # prices=np.array([price1, price2,price3,price4,price5,price6,price7])    
    # input_df=pd.DataFrame(index=np.arange(1,8), columns=['Price'])
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

#    recommendation, results, query=recommend_today()   
    query=pd.read_csv('tradepal/models/query.csv', sep=',', index_col=0,delimiter=None, header='infer')

   
    
    
    result=""
    if query.loc[symbol,mod]==0:
        result='BUY'
    elif query.loc[symbol,mod]==2:
        result='SELL'
    else:
        result='HOLD'
        
    "   "
    "   "
    

    if st.button("Predict"):
        # result=predict_note_authentication(variance,skewness,curtosis,entropy)
        # result
        # st.success('The output is: {}'.format(result))
        st.success(f'### The predicted trading for today is:  {result}')
        # st.write(f"classifier={classifier_name}")
    "   "
    "   "
    final=query.loc[symbol,'recommend']
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
    # df = data_module.data
    
    st.title('Index Fund Data Analysis')
    
    st.write("### Historical normalized prices variations for the 5 representative index funds is shown below:")
    st.image(load_image('tradepal/resources/price_variations.png'), use_column_width=True)
    
    "   "
    st.write('### Below we explore the SPY fund index stock data in 2011.')
    
    symbol='SPY'		
#    sd=dt.datetime(2010,1,1)
#    ed=dt.datetime(2010, 12, 31)   	  			  	 		  		  		    	 		 		   		 		  
#    dates = pd.date_range(sd, ed)  		
    df_dataX, df_dataY, indicators=get_XY_data(symbol, sd=dt.datetime(2010,1,1), ed=dt.datetime(2010, 12, 31),impact=0.0)   	  			  	 		  		  		    	 		 		   		 		  
#    prices_all = get_data([symbol], dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
#    prices = prices_all[[symbol,]]  # only portfolio symbol  	
#    # prices_norm=normalize_prices(prices)	
#    
#    learner=sl.StrategyLearner(verbose = False, impact=0.0)# constructor
#    learner.addEvidence(symbol = "SPY", sd=dt.datetime(2011,1,1), ed=dt.datetime(2011,12,31), sv = 100000)# training phase
#    test_trades, testX, test_dataY=learner.testPolicy(symbol = "SPY", sd=dt.datetime(2011,1,1), ed=dt.datetime(2011,12,31), sv = 100000)
#    # predTrade_df=learner.predTrade(input_df, symbol = "SPY",  sv = 100000)
    
    
    # Dataframe samples
    st.subheader("Sample of raw dataset")
    raw=pd.read_csv("tradepal/data/SPY.csv")     
    st.write(raw.iloc[176:185,])
    st.subheader("Features used in training")
    st.dataframe(df_dataX[30:35])
    st.write("Note: SMA=simple moving average, BB_pct= Bollinger Band percent, CCI=Commodity Channel Index, OBV=On-Balance Volume, macd=moving average convergence divergence, rsi_12=Relative strength index (window is 12), kdjj=stochastic oscillator, adx=Average directional index   ")
#    st.table(df_dataX[30:35])
    # Description length histogram
    st.subheader("Trading performance of different methods measured by normalized market value")
    # hist_values = np.histogram(
    # df['description'].str.len().tolist(), bins=24)[0]
    # st.bar_chart(hist_values)
    
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
    
    
 #get trade dataframe
    # test_trades, test_dataY=learner.testPolicy(symbol = "SPY", sd=dt.datetime(2010,1,1), ed=dt.datetime(2010,12,31), sv = 100000)
    # ####pred_trade
    
    # sl_portvals=compute_portvals(test_trades,start_val = 100000, commission=0.00, impact=0.00)    
    # cum_ret1, ave_daily_ret1, daily_ret_std1, sharpe_ratio1=compute_port_stats(sl_portvals)
    
    # print("  ")
    # print('Strategy Learner Stats:')
    # print("cumulative return: " + str(cum_ret1))
    # print("Avg of daily returns: " + str(ave_daily_ret1))
    # print("Std deviation of daily returns " + str(daily_ret_std1))
    # print("Sharpe Ratio " + str(sharpe_ratio1))
    # print("  ")    
    
    # prices_norm=normalize_prices(input_df['price'])
    # skewness = st.number_input("skewness")
    # curtosis = st.number_input("curtosis")
    # entropy = st.number_input("entropy")


    
    
#app=Flask(__name__)
#Swagger(app)

# pickle_in = open("classifier.pkl","rb")
# classifier=pickle.load(pickle_in)

# #@app.route('/')
# def welcome():
#     return "Welcome All"

# #@app.route('/predict',methods=["Get"])
# def predict_note_authentication(variance,skewness,curtosis,entropy):
    
#     """Let's Authenticate the Banks Note 
#     This is using docstrings for specifications.
#     ---
#     parameters:  
#       - name: variance
#         in: query
#         type: number
#         required: true
#       - name: skewness
#         in: query
#         type: number
#         required: true
#       - name: curtosis
#         in: query
#         type: number
#         required: true
#       - name: entropy
#         in: query
#         type: number
#         required: true
#     responses:
#         200:
#             description: The output values
        
#     """
   
#     prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
#     print(prediction)
#     return prediction
    

# DATE_TIME = "date/time"
# DATA_URL = (
#     "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
# )

# st.title('Streamlit example')

# dataset_name= st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine dataset"))
# st.write(dataset_name)
# classifier_name= st.sidebar.selectbox("Select classifier",("KNN","SVM","Random forest"))

# def get_dataset(dataset_name):
#     if dataset_name=="Iris":
#         data=datasets.load_iris()
#     elif dataset_name=="Breast Cancer":
#         data=datasets.load_breast_cancer()
#     else:
#         data=datasets.load_wine()
#     X=data.data
#     y=data.target
#     return X,y

# # X, y=get_dataset(dataset_name)
# # X,y
# X, y=get_dataset(dataset_name)
# st.write("shape of dataset",X.shape)
# st.write("number of classes", len(np.unique(y)), np.unique(y))
# st.write(dataset_name)

# def add_parameter_ui(clf_name):
#     params=dict()
#     if clf_name=="KNN":
#         K=st.sidebar.slider("K",1,15)
#         params["K"]=K
#     elif clf_name=="SVM":
#         C=st.sidebar.slider("C",0.01,10.0)
#         params["C"]=C
#     elif clf_name=="Random forest":
#         max_depth=st.sidebar.slider("max_depth",2,15)
#         n_estimators=st.sidebar.slider("n_estimators",1,100)
#         params["max_depth"]=max_depth
#         params["n_estimators"]=n_estimators
#     return params

# params=add_parameter_ui(classifier_name)

# def get_classifier(clf_name,params):
#     if clf_name=="KNN":
#         clf=KNeighborsClassifier(n_neighbors=params["K"])
#     elif clf_name=="SVM":
#         clf=SVC(C=params["C"])
#     elif clf_name=="Random forest":
#         clf=RandomForestClassifier(max_depth=params["max_depth"],n_estimators=params["n_estimators"], random_state=12345)        
#     return clf

# clf=get_classifier(classifier_name,params)

# #Classification
# X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=12345)
# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)
# acc=accuracy_score(y_test, y_pred)
# st.write(f"classifier={classifier_name}")
# st.write(f"accuracy = {acc}")

# #PLOT
# pca=PCA(2)
# X_projected=pca.fit_transform(X)
# x1=X_projected[:,0]
# x2=X_projected[:,1]

# fig=plt.figure()
# plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
# plt.xlabel("Principle Component 1")
# plt.ylabel("Principle Component 2")
# plt.colorbar()
# # plt.show()
# st.pyplot()






# TODO
# add more parameters
# add other classifier
# add feature scaling

# st.write("## Uber Pickups in New York City")
# st.markdown(
# """
# This is a demo of a Streamlit app that shows the Uber pickups
# geographical distribution in New York City. Use the slider
# to pick a specific hour and look at how the charts change.
# [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
# """)

# @st.cache(persist=True)
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis="columns", inplace=True)
#     data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
#     return data


# data = load_data(100000)
# # hour=8
# #hour=st.sidebar.slider('hour',0,23,10)
# hour=st.sidebar.number_input('hour',0,23,10)
# data=data[data[DATE_TIME].dt.hour==hour]

# #'# Raw data at %shr' %hour, data

# '## Geo Data at %s clock' % hour
# #st.map(data)
# st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24))
# midpoint = (np.average(data["lat"]), np.average(data["lon"]))

# st.write(pdk.Deck(
#     map_style="mapbox://styles/mapbox/light-v9",
#     initial_view_state={
#         "latitude": midpoint[0],
#         "longitude": midpoint[1],
#         "zoom": 11,
#         "pitch": 50,
#     },
#     layers=[
#         pdk.Layer(
#             "HexagonLayer",
#             data=data,
#             get_position=["lon", "lat"],
#             radius=100,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             pickable=True,
#             extruded=True,
#         ),
#     ],
# ))

# if st.checkbox('show raw data:'):
#     '## Raw data at %s clock' %hour, data
