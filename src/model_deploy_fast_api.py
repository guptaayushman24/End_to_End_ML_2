import uvicorn
from fastapi import FastAPI
from src.Input_Parameter import CarPricePrediction
import pandas as pd
import numpy as np
import pickle
# Create the app object
app = FastAPI()
pickle_in = open('artifacts\model.pkl','rb')
regressor = pickle.load(pickle_in)

@app.get('/')
def index() :
    return {'message':'Hello, World'}
@app.get('/{name}')
def get_name(name:str) :
    return {'Welcome to car prediction website'}

@app.post('/predict')
def predict_car_price(data:CarPricePrediction) :
    data = data.dict()
    print(data)
    curbweight = data['curbweight']
    enginesize = data['enginesize']
    horsepower = data['horsepower']
    citympg = data['citympg']
    highwaympg = data['highwaympg']

    prediction = regressor.predict([[curbweight,enginesize,horsepower,citympg,highwaympg]])
    prediction_json = prediction.tolist()

    return "The predicted car price is {}".format(prediction_json)



