# Databricks notebook source
import mlflow
from mlflow import MlflowClient

# COMMAND ----------

model_name = "ml_iris"
stage = "Staging"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

# COMMAND ----------

def get_latest_staging_model(model_name):

  """Go through all the model versions and find the model thats in staging"""
  client = MlflowClient()
  model_versions = client.search_model_versions(f"name='{model_name}'")
    
  staging_model_versions = [mv for mv in model_versions if mv.current_stage == 'Staging']

  latest_staging_model = staging_model_versions[0]
  return latest_staging_model

# Using the function
model_name = "ml_iris"
latest_model_version = get_latest_staging_model(model_name)
# latest_model_version.version

# COMMAND ----------

url = "https://adb-984752964297111.11.azuredatabricks.net/api/2.0/serving-endpoints"
headers = {"Authorization": "Bearer dapiacb48ee7d1525f65280bc8302737131c"}
endpoint_name ="iris-sg-test"
payload = {
  "name": "iris-sg-test",
  "config":{
   "served_models": [{
     "model_name": "ml_iris",
     "model_version": latest_model_version.version,
     "workload_size": "Small",
     "scale_to_zero_enabled": True
    }]
  }
}


# COMMAND ----------

def get_endpoint(endpoint_name:str=endpoint_name, headers:dict=headers) -> str:
  response = requests.request("GET", f"{url}/{endpoint_name}",headers=headers)
  return response.text

def update_endpoint(endpoint_name:str=endpoint_name, headers:dict=headers, payload:dict=payload) -> str:
  response = requests.request("PUT", f"{url}/{endpoint_name}/config",headers=headers, data=json.dumps(payload))
  return response.text


def deploy_endpoint(endpoint_name:str=endpoint_name, headers:dict=headers, payload:dict=payload) -> str:
  response = requests.request("POST", f"{url}",headers=headers, data=json.dumps(payload))
  return response.text

def delete_endpoint(endpoint_name:str=endpoint_name, headers:dict=headers) -> str:
  response = requests.request("DELETE", f"{url}/{endpoint_name}",headers=headers)


# COMMAND ----------

delete_endpoint()

# COMMAND ----------


current_deployed_model = get_endpoint()
if json.loads(current_deployed_model).get('name') == endpoint_name:
  pprint.pprint("INFO: Updating Endpoint")
  update_endpoint()
else:
  pprint.pprint("INFO: Deploying New Endpoint")
  deploy_endpoint()


# COMMAND ----------

import time
import pprint
time.sleep(10)
current_deployed_model = get_endpoint()
pprint.pprint(json.loads(current_deployed_model))
