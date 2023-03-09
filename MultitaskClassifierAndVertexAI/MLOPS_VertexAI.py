# Databricks notebook source
# MAGIC %pip install google-cloud-mlflow

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes 
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# COMMAND ----------

# load dataset
db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
 
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()
 
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run() as run:  
  # Set the model parameters. 
  n_estimators = 100
  max_depth = 6
  max_features = 3
  # Create and train model.
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  # Use the model to make predictions on the test dataset.
  predictions = rf.predict(X_test)
  
mlflow.end_run()

# COMMAND ----------

model_name = "vertex-sklearn-blog-demo-samaya"
mlflow.sklearn.log_model(rf, model_name, registered_model_name=model_name)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_version_infos = client.search_model_versions(f"name = '{model_name}'")
model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
model_uri=f"models:/{model_name}/{model_version}"

# model_uri should be models:/vertex-sklearn-blog-demo/1

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
