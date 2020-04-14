from sklearn.externals import joblib
import numpy as np
import pandas as pd
import os
from io import StringIO, BytesIO

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    if request_content_type == "application/python-pickle":
        array = np.load(BytesIO(request_body), allow_pickle=True)
        return array
    else:
        raise Exception("Please provide 'application/python-pickle' as a request content type")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return np.array(prediction).astype(int)

