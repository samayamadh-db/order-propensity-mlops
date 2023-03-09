# Databricks notebook source
# MAGIC %pip install tensorflow-addons
# MAGIC %pip install tabtransformertf

# COMMAND ----------

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep


# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC plt.rcParams["figure.figsize"] = (20,10)
# MAGIC plt.rcParams.update({'font.size': 15})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Data

# COMMAND ----------

CSV_HEADER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]

train_data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
train_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)

test_data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")

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

display(train_data_orders)

# COMMAND ----------

print(f"Train dataset shape: {train_data_orders.shape}")
print(f"Test dataset shape: {test_data_orders.shape}")

# COMMAND ----------

print(train_data_orders.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess

# COMMAND ----------

# Column information
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = ['basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by', 'image_picker', 'account_page_click', 'promo_banner_click', 'detail_wishlist_add', 'list_size_dropdown', 'closed_minibasket_click', 'checked_delivery_detail', 'checked_returns_detail', 'sign_in', 'saw_checkout', 'saw_sizecharts', 'saw_delivery', 'saw_account_upgrade', 'saw_homepage', 'device_mobile', 'device_computer', 'device_tablet', 'returning_user', 'loc_uk']

FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
LABEL = 'ordered'

# COMMAND ----------

# encoding as binary target
train_data_orders[LABEL] = train_data_orders[LABEL].apply(lambda x: int(x == '0')) 
test_data_orders[LABEL] = test_data_orders[LABEL].apply(lambda x: int(x == '0'))
train_data_orders[LABEL].mean(), test_data_orders[LABEL].mean()

# COMMAND ----------

test_data_orders = test_data_orders.iloc[1:, :] # drop invalid row

# COMMAND ----------

# Set data types
train_data_orders[CATEGORICAL_FEATURES] = train_data_orders[CATEGORICAL_FEATURES].astype(str)
test_data_orders[CATEGORICAL_FEATURES] = test_data_orders[CATEGORICAL_FEATURES].astype(str)

train_data_orders[NUMERIC_FEATURES] = train_data_orders[NUMERIC_FEATURES].astype(float)
test_data_orders[NUMERIC_FEATURES] = test_data_orders[NUMERIC_FEATURES].astype(float)

# COMMAND ----------

# Train/test split
X_train, X_val = train_test_split(train_data_orders, test_size=0.2)

# COMMAND ----------

sc = StandardScaler()
X_train.loc[:, NUMERIC_FEATURES] = sc.fit_transform(X_train[NUMERIC_FEATURES])
X_val.loc[:, NUMERIC_FEATURES] = sc.transform(X_val[NUMERIC_FEATURES])
test_data_orders.loc[:, NUMERIC_FEATURES] = sc.transform(test_data_orders[NUMERIC_FEATURES])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling Prep

# COMMAND ----------

train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL)
val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
test_dataset = df_to_dataset(test_data_orders[FEATURES + [LABEL]], shuffle=False) # No target, no shuffle

# COMMAND ----------

# MAGIC %md
# MAGIC # FTTransformer

# COMMAND ----------

# MAGIC %md
# MAGIC ## FT Transformer - Linear Numerical Encoding

# COMMAND ----------

ft_linear_encoder = FTTransformerEncoder(
    numerical_features = NUMERIC_FEATURES,
    categorical_features = CATEGORICAL_FEATURES,
    numerical_data = X_train[NUMERIC_FEATURES].values,
    categorical_data = X_train[CATEGORICAL_FEATURES].values,
    y = None,
    numerical_embedding_type='linear',
    embedding_dim=16,
    depth=4,
    heads=8,
    attn_dropout=0.2,
    ff_dropout=0.2,
    explainable=True
)

# Pass the encoder to the model
ft_linear_transformer = FTTransformer(
    encoder=ft_linear_encoder,
    out_dim=1,
    out_activation='sigmoid',
)

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 50

optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

ft_linear_transformer.compile(
    optimizer = optimizer,
    loss = {"output": tf.keras.losses.BinaryCrossentropy(), "importances": None},
    metrics= {"output": [tf.keras.metrics.AUC(name="PR AUC", curve='PR')], "importances": None},
)

early = EarlyStopping(monitor="val_output_loss", mode="min", patience=20, restore_best_weights=True)
callback_list = [early]

ft_linear_history = ft_linear_transformer.fit(
    train_dataset, 
    epochs=NUM_EPOCHS, 
    validation_data=val_dataset,
    callbacks=callback_list
)

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

test_data.income_bracket.unique()

# COMMAND ----------

linear_test_preds = ft_linear_transformer.predict(test_dataset)
print("FT-Transformer with Linear Numerical Embedding")
print("Test ROC AUC:", np.round(roc_auc_score(test_data[LABEL], linear_test_preds['output'].ravel()), 4))
print("Test PR AUC:", np.round(average_precision_score(test_data[LABEL], linear_test_preds['output'].ravel()), 4))
print("Test Accuracy:", np.round(accuracy_score(test_data[LABEL], linear_test_preds['output'].ravel()>0.5), 4))

# Reported accuracy - 0.858

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explainability

# COMMAND ----------

linear_importances = linear_test_preds['importances']
linear_importances_df = pd.DataFrame(linear_importances[:, :-1], columns = FEATURES)
linear_total_importances = get_model_importances(
    linear_importances_df, title="Importances for FT-Transformer with Linear Numerical Embedddings"
)

# COMMAND ----------

# Largest prediction
max_idx = np.argsort(linear_test_preds['output'].ravel())[-1]
example_importance_linear = linear_importances_df.iloc[max_idx, :].sort_values(ascending=False).rename("Importance").to_frame().join(
    test_data.iloc[max_idx, :].rename("Example Vlaue")
).head(5)
print(f"Top 5 contributions to row {max_idx} which was scored {str(np.round(linear_test_preds['output'].ravel()[max_idx], 4))}")
display(example_importance_linear)

# Smallest one
min_idx = np.argsort(linear_test_preds['output'].ravel())[0]
example_importance_linear = linear_importances_df.iloc[min_idx, :].sort_values(ascending=False).rename("Importance").to_frame().join(
    test_data.iloc[min_idx, :].rename("Example Vlaue")
).head(5)
print(f"Top 5 contributions to row {min_idx} which was scored {str(np.round(linear_test_preds['output'].ravel()[min_idx], 4))}")
display(example_importance_linear)

# COMMAND ----------


