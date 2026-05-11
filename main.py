from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import pandas as pd

app = FastAPI(title="API de classification de fruits avec KNN")
model = joblib.load("model_knn.pkl")
encoder = joblib.load("encoder.pkl")
class Fruit(BaseModel):
    poids: float
    diametre: float
    couleur: float
    
@app.get("/")
def Accueil():
    return {"message": "Bienvenue sur l'API de classification de fruits avec KNN!"}

@app.post("/predict")
def predict(fruit: Fruit):
   data = pd.DataFrame([{"poids": fruit.poids, "diametre": fruit.diametre, "couleur": fruit.couleur}])
   prediction = model.predict(data)
   predicted_label = encoder.inverse_transform(prediction)[0]
   return {"predicted_fruit": predicted_label}
@app.get("/health")
def health_check():
    return {"status": "sante de l'API", "model charge": True if model else False}