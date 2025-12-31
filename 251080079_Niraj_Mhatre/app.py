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

## Question 3: Backend Execution and Testing

#Start the backend server
The backend server was started using the Uvicorn command:
uvicorn app:app --reload

# Access the API
The API was accessed at:
http://127.0.0.1:8000

# Send sample input
Sample input data was sent using a browser/Postman.

#Verify output
The API returned the predicted traffic speed in JSON format.
