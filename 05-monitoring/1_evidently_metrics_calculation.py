import datetime
import time
import random
import logging 
import pandas as pd
import psycopg
import joblib

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, DataQualityStabilityMetric, ColumnQuantileMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 1
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	number_not_stable_prediction integer,
	current_data_quantile float
)
"""

year = 2024
month = 3
taxi_type = 'green'

print("Start loading the data")
raw_data = pd.read_parquet(f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet")
print(f"Loaded the {taxi_type} taxi trip data for {year:04d}-{month:02d}. Rows: {raw_data.shape[0]}")

reference_data = pd.read_parquet('data/reference.parquet')
print(f"Loaded the reference data. Rows: {reference_data.shape[0]}")

with open('models/lin_reg.bin', 'rb') as f_in:
	model = joblib.load(f_in)
print("Loaded the model")

begin = datetime.datetime(2024, 3, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    DataQualityStabilityMetric(),
    ColumnQuantileMetric(column_name='fare_amount', quantile=0.5)
])

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, i):
	current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
		(raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

	#current_data.fillna(0, inplace=True)
	current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

	report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
	number_not_stable_prediction = result['metrics'][3]['result']['number_not_stable_prediction']
	current_data_quantile = result['metrics'][4]['result']['current']['value']

	curr.execute(
		"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, number_not_stable_prediction, current_data_quantile) values (%s, %s, %s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values, number_not_stable_prediction, current_data_quantile)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 27):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=1)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()
