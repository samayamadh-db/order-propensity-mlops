# Databricks notebook source
# MAGIC %md ###Step 1: Setup the Environment
# MAGIC
# MAGIC There are additional Python libraries which you will need to install and attach to your cluster in the %pip cell below. 

# COMMAND ----------

# MAGIC %pip install google-cloud-mlflow
# MAGIC %pip install pytorch-tabnet

# COMMAND ----------

import math
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from pytorch_tabnet.tab_model import TabNetClassifier


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
# MAGIC FTTransformer  - Linear Numerical Encoding

# COMMAND ----------




# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()
 
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run() as run:  
    clf = TabNetClassifier()
    clf.fit(X_train.to_numpy(), y_train.to_numpy(), max_epochs = 2)  
mlflow.end_run()


# COMMAND ----------

preds = clf.predict(X_test.to_numpy())

# COMMAND ----------

print(preds)

# COMMAND ----------

print(y_test.to_numpy())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Evaluation 

# COMMAND ----------

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].plot(ft_linear_history.history['loss'], label='Training Loss')
ax[0].plot(ft_linear_history.history['val_loss'], label='Validation Loss')
ax[0].legend()

ax[1].plot(ft_linear_history.history['output_PR AUC'], label='Training PR AUC')
ax[1].plot(ft_linear_history.history['val_output_PR AUC'], label='Validation PR AUC')
ax[1].legend()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Package model and log it to mlflow 

# COMMAND ----------

# Define the model class
class TabTransformerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, n):
      self.tabt_model = n
      
    def predict(self, context,  model_input):
        return self.tabt_model.predict(model_input)

# COMMAND ----------

# add tabtransformertf and tensorflow-addons to conda environment info
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'] += ['pytorch-tabnet==4.0'] 
conda_env['dependencies'][2]['pip'] += ['torch==1.9.0+cpu'] 


# COMMAND ----------

conda_env

# COMMAND ----------

mlflow_pyfunc_model_path = "orderpropensity_tabtrans"
# save model run to mlflow
with mlflow.start_run(run_name='tab_transformer_samaya_run') as run:
 mlflow.pyfunc.log_model(mlflow_pyfunc_model_path,python_model=TabTransformerWrapper(clf), conda_env=conda_env, input_example=X_test.to_numpy())


# COMMAND ----------

X_test.to_numpy()

# COMMAND ----------

model_uri = "runs:/{0}/{1}".format(run.info.run_id,mlflow_pyfunc_model_path)
print(model_uri)
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
loaded_model.predict(pd.DataFrame(X_test).to_numpy())

# COMMAND ----------

linear_test_preds = loaded_model.predict(X_test.to_numpy())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Push model to Model Registry 

# COMMAND ----------

deploy_name = 'orderpropensity_tabtrans_model'
mlflow.register_model(model_uri, deploy_name)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_version_infos = client.search_model_versions(f"name = '{deploy_name}'")
model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
deploy_uri=f"models:/{deploy_name}/{model_version}"
print(deploy_uri)

# COMMAND ----------

model = mlflow.pyfunc.load_model(deploy_uri)
model.predict(X_test.to_numpy())

# COMMAND ----------

model_name = "tabnet-multitask-classifier-samaya"
mlflow.sklearn.log_model(clf, model_name, registered_model_name=model_name)

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
mlflow.register_model(model_uri=model_uri, name="smadhavan_tabtrans_from_gcp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Deploy model for real time serving via VertexAI 

# COMMAND ----------

# Really simple Vertex client instantiation
vtx_client = mlflow.deployments.get_deploy_client("google_cloud")
# Deploy to Vertex AI using three lines of code! Note: If using python > 3.7, this may take up to 20 minutes.
deployment = vtx_client.create_deployment(
    name=deploy_name,
    model_uri=deploy_uri,config= dict(machine_type= "n1-standard-2",
            min_replica_count=1,
            max_replica_count=1,
            endpoint_deploy_timeout=18000))

# COMMAND ----------

vtx_client.list_deployments()

# COMMAND ----------

# Use the .predict() method from the same plugin
predictions = vtx_client.predict(deploy_name, X_test.to_numpy())
print(predictions)
