# TradePal
#### A Free, Transparent and Efficient Stock Trading Assistant Powered by Machine Learning

## Motivation for this project:
As we all know, the stock market is very risky, at the same time the professional wealth management services are pricey, thus regular non-professional stockholders face great risks in stock trading. This project addresses this headache by providing a free, transparent and efficient web app powered by machine learning to help non-professionals on stock trading, by providing the predictions of today's optial trading option (Buy, Sell or Hold) as well as final recommendation. In addition, the app provides backtesting results in the past 3 years from all the 5 machine learning models such that the user can make their own trading decision based on those results. 

The figure below presents a bried demo of the TradePal web app.

![](app_demo.gif)

## Structure of this repository
All the data, sources codes and related resources are included in the directory `tradpal`, and the contents of each subdirectory are described below.<br />
`1) data:` all the related source data (e.g., historical stock prices, historical US dollar LIBOR interest rates) reside here.  <br /><br />
`2) models:` all the source codes for the 5 machine learning models, and the well-trained models and associated parameters reside here. The source codes for the lstm model are in the file `lstm.py` and lstm is trained separately using `lstm_grid_search.py` to seek the best hyperparameters. Source codes for all 4 other models reside in the `models.py` file. The file `recommend.py` is used to provides the fund- and model-specific prediction for today's trading option and the app's final recommendation as well.<br /><br />
`3) resources:` all supporting images and auxillary resources for the app reside here.  <br /><br />



###



## Setup
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


## Run Streamlit App
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

## Deploy to Google Kubernetes Engine (GKE)
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



