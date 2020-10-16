# TradePal
#### A Free, Transparent and Efficient Stock Trading Assistant (Web App) Powered by Machine Learning

## Motivation:
As we all know, the stock market is very risky, at the same time the professional wealth management services are pricey, thus regular non-professional stockholders face great risks in stock trading. This project addresses this headache by providing a free, transparent and efficient web app powered by machine learning to help non-professionals improve their performances on stock trading. 

## Functionalities:
The TradePal web app offers trading predictions, recommendations and relevant results for 5 representative index funds that track major stock indices:
- SPY (SP 500)
- DIA (Dow Jones Industrial Average)
- QQQ (Nasdaq 100),
- TLT (U.S. Treasury 20+ Year Bond)
- IWM (Russell 2000)<br />

The app is powered by 5 widely-used machine learning models:
- Logistic Regression
- Random Forest
- Ada Boosting 
- Support Vector Machine
- Long Short-Term Memory<br />

Specifically, it has the following functionalities:
- provides model-specific prediction of today's optial trading option (Buy, Sell or Hold) by each of the 5 machine learning models.
- provides final recommendation for today's optial trading option. 
- provides backtesting results in the past 3 years from all the 5 models such that the user can make their own trading decision based on those results. 

The figure below presents a bried demo of the TradePal web app.

![](app_demo.gif)

## Structure of this repository
All the data, sources codes and related resources are included in the directory `tradpal`, and the contents of each subdirectory are described below.<br />
- `data:` all the related source data (e.g., historical stock prices, historical US dollar LIBOR interest rates) reside here.  <br /><br />
- `models:` all the well-trained models and associated parameters reside here. <br /><br />
- `resources:` all supporting images and auxillary resources for the app reside here.  <br /><br />
- `src:` all the source codes for the 5 machine learning models reside here, which is rendered in object oriented programming (OOP). The hyperparameters of the lstm model are tuned separately using the codes in `lstm_grid_search.py`. The file `recommend.py` is used to provides the fund- and model-specific prediction for today's trading option and the app's final recommendation as well.



###



## App Setup
Clone repository
```
git clone https://github.com/yalingliupnnl/TradePal.git
cd ./TradePal
```

## Requisites
#### Dependencies
- [Anaconda] (https://docs.anaconda.com/anaconda/install/)
- [Streamlit](streamlit.io)

#### Installation
To install the package above, please run:
```shell
conda create --name tradepal python=3.7
conda activate tradepal
pip install -r requirements.txt
```


## Run TradePal App
```
streamlit run app.py
```
Optional: Docker build
```
docker build -t tradepal-streamlit:v1 -f Dockerfile.app .
docker run -p 8501:8501 tradepal-streamlit:v1
```

<!-- ## Train Model
The config.yaml file contains the final mode parameters for input into the training script. 
```
cd train
pip install -r requirements.txt
python3 train.py -y './config.yaml'
```
Optional: Docker build
```
docker build -t cloudwine-train:v1 -f Dockerfile.train . 
```-->

## Deploy TradePal to Google Kubernetes Engine (GKE)
Based off the intruction from Google's 'Deploying a containerized web application' (https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app).

Prerequisites:
1) A Google Cloud (GC) Project with billing enabled.
2) The GC SDK installed (https://cloud.google.com/sdk/docs/quickstarts)
3) Install kubernetes
```
gcloud components install kubectl
```

Set up gcloud tool (GC SDK)
```
export PROJECT_ID = gcp-project-name
export ZONE = gcp-compute-zone (e.g. us-westb-1)

gcloud config set project $PROJECT_ID
gcloud config set compute/zone compute-zone

gcloud auth configure-docker
```

Build and push the container image to GC Container Registery:
```
docker build -t gcp.io/$(PROJECT_ID}/tradepal-streamlit:v1 -f Dockerfile.app .
docker push gcr.io/${PROJECT_ID}/tradepal-streamlit:v1
```

Create GKE Cluster
```
gcloud container clusters create tradepal-cluster --machine-type=n1-highmem-2
gcloud compute instances list
```

Deploy app to GKE
```
kubectl create deployment tradepal-app --image=gcr.io/${PROJECT_ID}/tradepal-app:v1
kubectl autoscale deployment tradepal-app --cpu-percent=80 --min=1 --max=5
kubectl get pods
```

Expose app to internet
```
kubectl expose deployment tradepal-app --name=tradepal-app-service --type=LoadBalancer --port 80 --target-port 8080
kubectl get service
```

---

Deleting the deployment
```
kubectl delete service cloudwine-app-service
gcloud container clusters delete cloudwine-cluster
```



## Data Exploration
Run the streamlit app and see the 'Data Exploration' page for data exploration and experiment results. Below is an example of the performance comparison between Random Forest and the Benchmark strategy (buy and hold) for SPY (SP 500 index fund). 

![](tradepal/resources/SPY_training.png?raw=true)



