import boto3
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score
import json

client = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME') 
CONTENT_TYPE = 'application/python-pickle'

test_data = pd.read_csv('./models/test.csv', header=None, names=None)
test_y = test_data.iloc[:,0]
test_x = test_data.iloc[:, 1:]

response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType=CONTENT_TYPE,
        Body=pickle.dumps(test_x)
        )

test_y_preds = json.loads(response['Body'].read().decode('utf-8'))

print('Accuracy Score: ', accuracy_score(test_y, test_y_preds))
