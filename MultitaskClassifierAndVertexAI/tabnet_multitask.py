# Databricks notebook source
# MAGIC %pip install pytorch-tabnet

# COMMAND ----------

# MAGIC %pip install google-cloud-mlflow

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# COMMAND ----------

# load dataset
from sklearn.datasets import load_diabetes 
db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# COMMAND ----------



# COMMAND ----------

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

 
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()
 
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run() as run:  
    clf = TabNetClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
  
mlflow.end_run()


# COMMAND ----------

model_name = "tabnet-multitask-classifier-samaya"
mlflow.sklearn.log_model(clf, model_name, registered_model_name=model_name)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_version_infos = client.search_model_versions(f"name = '{model_name}'")
model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
model_uri=f"models:/{model_name}/{model_version}"



# COMMAND ----------

print(model_uri)

# COMMAND ----------

# Really simple Vertex client instantiation
vtx_client = mlflow.deployments.get_deploy_client("google_cloud")
deploy_name = f"{model_name}-{model_version}"

# Deploy to Vertex AI using three lines of code! Note: If using python > 3.7, this may take up to 20 minutes.
deployment = vtx_client.create_deployment(
    name=deploy_name,
    model_uri=model_uri)

# COMMAND ----------


# Use the .predict() method from the same plugin
predictions = vtx_client.predict(deploy_name, X_test)

# COMMAND ----------

print(predictions)

# COMMAND ----------


