#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression

mar_data = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-03.parquet')

mar_data.shape[0]

print("create target")
mar_data["duration_min"] = mar_data.lpep_dropoff_datetime - mar_data.lpep_pickup_datetime
mar_data.duration_min = mar_data.duration_min.apply(lambda td : float(td.total_seconds())/60)

print("filter out outliers")
mar_data = mar_data[(mar_data.duration_min >= 0) & (mar_data.duration_min <= 60)]
mar_data = mar_data[(mar_data.passenger_count > 0) & (mar_data.passenger_count <= 8)]

print("data labeling")
target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

train_data = mar_data[:30000]
val_data = mar_data[30000:]

print("train a model")
model = LinearRegression()
model.fit(train_data[num_features + cat_features], train_data[target])

train_preds = model.predict(train_data[num_features + cat_features])
train_data['prediction'] = train_preds

val_preds = model.predict(val_data[num_features + cat_features])
val_data['prediction'] = val_preds


print("dump the model and reference data")
for dirname in ["models", "data"]:
    os.makedirs(os.path.join(os.getcwd(), dirname), exist_ok=True)
with open('models/lin_reg.bin', 'wb') as f_out:
    dump(model, f_out)
val_data.to_parquet('data/reference.parquet')
