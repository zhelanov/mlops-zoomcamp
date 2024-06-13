#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import os
import sys

year = int(sys.argv[1])
month = int(sys.argv[2])
taxi_type = 'yellow'

categorical = ['PULocationID', 'DOLocationID']
model_path = 'model.bin'
output_file = "result.parquet"

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def get_model(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

def get_prediction(dv, model, df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    return model.predict(X_val)


if __name__ == "__main__":
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet')
    dv, model = get_model(model_path)
    y_pred = get_prediction(dv, model, df)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print(f"mean predicted duration for {taxi_type} {year:04d}.{month:02d} trip data: {df_result['predicted_duration'].mean()}")