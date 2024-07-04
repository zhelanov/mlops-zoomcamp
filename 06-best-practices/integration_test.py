import os
import batch
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_saving(path):
  data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
  columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
  df = pd.DataFrame(data, columns=columns)
  os.environ['S3_ENDPOINT_URL']="http://localhost:4566"
  batch.save_data(path, df)


def test_reading(path):
  os.environ['INPUT_FILE_PATTERN'] = path
  batch.main(2023, 1)
  df = batch.read_data(path)
  predicted_durations = df['predicted_duration'].sum()
  print(f"the sum of predicted durations: {predicted_durations:.2f}")

if __name__ == "__main__":
  path = "s3://mybucket/yellow/2023/01/predictions.parquet"
  test_saving(path)
  test_reading(path)