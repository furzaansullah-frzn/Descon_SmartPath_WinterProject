# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 01:53:48 2025

@author: Niraj Mhatre
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app=FastAPI()

with open("model.pkl","rb") as f:
    model=pickle.load(f)

class InputData(BaseModel):
    features:list[float]

@app.post("/predict")
def predict(data:InputData):
    x=np.array(data.features).reshape(1,-1)
    y=float(model.predict(x)[0])
    return {"predicted_speed":y}
