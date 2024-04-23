from fastapi import APIRouter, Form
import numpy as np
import sklearn
import pickle

router = APIRouter(prefix="/cropPlanner")

with open('routers/CropPlanner.pkl', 'rb') as f:
    crop_planner = pickle.load(f)


crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']


@router.post("/predict")
async def crop_recommendation(N: float = Form(...), P: float = Form(...), K: float = Form(...),
                             temperature: float = Form(...), humidity: float = Form(...),
                             ph: float = Form(...), rainfall: float = Form(...)):
    # Create input features
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction
    predictions = crop_planner.predict(features)
    prediction = predictions[0]

    # Wrap prediction in a dictionary
    return {"prediction": crops[prediction]}