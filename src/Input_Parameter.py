from pydantic import BaseModel
class CarPricePrediction (BaseModel) :
    curbweight : float
    enginesize : float
    horsepower : float
    citympg : float
    highwaympg : float