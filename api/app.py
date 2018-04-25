from flask import Flask
from flask import request
from flask import json
import boto3
import pickle
# import os
# from sklearn.externals import joblib

BUCKET_NAME = 'www.abstract-significance-prediction.com'
MODEL_FILE_NAME = 'linearSVC_model_aws.pkl'

app = Flask(__name__)

S3 = boto3.client('s3' , region_name='US-east-2')

def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper

@app.route('/', methods=['POST'])
def index():
     # Parse request body for model input
    body_dict = request.get_json(silent=True)
    data = body_dict['data']

    prediction = predict(data)

    result = {'prediction': prediction}
    return json.dumps(result)

# https://s3.console.aws.amazon.com/s3/#

@memoize
def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=MODEL_FILE_NAME)
    model_str = response['Body'].read()

    model = pickle.loads(model_str)

    return model

def predict(data):
    model = load_model(MODEL_FILE_NAME)

    return model.predict(data).tolist()


if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)