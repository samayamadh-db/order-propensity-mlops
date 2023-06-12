# Databricks notebook source
# MAGIC %md ###Step 1: Setup the Environment
# MAGIC
# MAGIC There are additional Python libraries which you will need to install and attach to your cluster in the %pip cell below. 

# COMMAND ----------

# MAGIC %pip install google-cloud-mlflow
# MAGIC %pip install protobuf==3.20.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import math
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


import mlflow
import mlflow.pyfunc
from mlflow.deployments import get_deploy_client

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 2: Download Data

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/samaya_ml/training_sample.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
train_data_orders = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location).toPandas()
#train_data_orders.withColumn('customer_average_spend',  np.random.randint(0,10000, size=train_data_orders.count()))
display(train_data_orders)

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/samaya_ml/testing_sample.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
test_data_orders = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location).toPandas()

display(test_data_orders)

# COMMAND ----------

print(train_data_orders.columns)

# COMMAND ----------

display(test_data_orders)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Data Preprocessing

# COMMAND ----------

# Column information
CATEGORICAL_FEATURES = ['basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by', 'image_picker', 'account_page_click', 'promo_banner_click', 'detail_wishlist_add', 'list_size_dropdown', 'closed_minibasket_click', 'checked_delivery_detail', 'checked_returns_detail', 'sign_in', 'saw_checkout', 'saw_sizecharts', 'saw_delivery', 'saw_account_upgrade', 'saw_homepage', 'device_mobile', 'device_computer', 'device_tablet', 'returning_user', 'loc_uk']
FEATURES =  list(CATEGORICAL_FEATURES)
LABEL = 'ordered'

# COMMAND ----------

# encoding as binary target
train_data_orders[LABEL] = train_data_orders[LABEL].apply(lambda x: int(x == '0')) 
test_data_orders[LABEL] = test_data_orders[LABEL].apply(lambda x: int(x == '0'))
train_data_orders[LABEL].mean(), test_data_orders[LABEL].mean()

# COMMAND ----------

test_data_orders.iloc[1:, :] # drop invalid row

# COMMAND ----------

# Set data types
train_data_orders[CATEGORICAL_FEATURES] = train_data_orders[CATEGORICAL_FEATURES].astype(int)
test_data_orders[CATEGORICAL_FEATURES] = test_data_orders[CATEGORICAL_FEATURES].astype(int)

# COMMAND ----------

train_data_orders = train_data_orders.drop('UserID',axis=1)
test_data_orders = test_data_orders.drop('UserID',axis=1)

# COMMAND ----------

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(train_data_orders,train_data_orders[LABEL], test_size=0.2, random_state=1)

# COMMAND ----------

X_train.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Model Training 
# MAGIC

# COMMAND ----------


mlflow.sklearn.autolog()

reg_rf = RandomForestClassifier()
with mlflow.start_run() as run: 
  reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)

# COMMAND ----------

from sklearn import metrics

print(metrics.classification_report(y_test, y_pred))

# COMMAND ----------

import mlflow
logged_model_uri = f"runs:/{run.info.run_id}/model"
 

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model_uri)


# COMMAND ----------


summaries = loaded_model.predict(X_test)

# COMMAND ----------

summaries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Push model to MLFlow Model Registry 

# COMMAND ----------

deploy_name = 'orderpropensity_sklearn_model'
mlflow.register_model(logged_model_uri, deploy_name)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_version_infos = client.search_model_versions(f"name = '{deploy_name}'")
model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
deploy_uri=f"models:/{deploy_name}/{model_version}"
print(deploy_uri)

# COMMAND ----------

model = mlflow.sklearn.load_model(deploy_uri)
model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Push model to Azure model registry

# COMMAND ----------

secret_scope = "feazure"
assert(len(dbutils.secrets.list(secret_scope))>0)
registry_secret_key_prefix = "feazure"

registry_uri = 'databricks://' + secret_scope + ':' + registry_secret_key_prefix if secret_scope and registry_secret_key_prefix else None

print("registry_uri: " + registry_uri)

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)
mlflow.register_model(model_uri=deploy_uri, name="smadhavan_sklearn_from_gcp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Deploy model for real time serving via VertexAI 

# COMMAND ----------

# Really simple Vertex client instantiation
vtx_client = mlflow.deployments.get_deploy_client("google_cloud")
# Deploy to Vertex AI using three lines of code! Note: If using python > 3.7, this may take up to 20 minutes.
deployment = vtx_client.create_deployment(
    name=deploy_name,
    model_uri=deploy_uri)

# COMMAND ----------

#vtx_client = mlflow.deployments.get_deploy_client("google_cloud")
vtx_client.list_deployments()

# COMMAND ----------

# Use the .predict() method from the same plugin
predictions = vtx_client.predict("orderpropensity_sklearn_model", X_test.head(100))
print(predictions)

# COMMAND ----------

type(X_test)

# COMMAND ----------


