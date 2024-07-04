import batch
from datetime import datetime
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_dummy():
  assert 1 == 1

def test_data_preparation():
  data = [
      (None, None, dt(1, 1), dt(1, 10)),
      (1, 1, dt(1, 2), dt(1, 10)),
      (1, None, dt(1, 2, 0), dt(1, 2, 59)),
      (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
  ]
  columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
  categorical = [columns[0], columns[1]]
  original_df = pd.DataFrame(data, columns=columns)
  prepared_df = batch.prepare_data(original_df, categorical)
  assert prepared_df.shape[0] == 2