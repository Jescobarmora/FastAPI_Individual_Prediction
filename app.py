from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pycaret.classification import predict_model
import json
import tempfile
import pandas as pd
import uvicorn
import pickle
import shutil
import traceback

path = "/code/Python/Corte_2/Quiz_2_2/Punto_1/"
file_name = 'predictions/predictions.json'

# Crear una instancia de FastAPI
app = FastAPI()

with open(path + 'models/ridge_model.pkl', 'rb') as file:
  modelo = pickle.load(file)


class InputData(BaseModel):
  Email: str
  Address: str
  Dominio: str
  Tecnologia: str
  Avg_Session_Length: float
  Time_on_App: float
  Time_on_Website: float
  Length_of_Membership: float


@app.get("/")
def home():
  # Retorna un simple mensaje de texto
  return 'Predicción estudiantes'


# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
  try:
    with open(path + file_name, 'r') as file:
      predictions = json.load(file)
  except (FileNotFoundError, json.JSONDecodeError):
    predictions = []

  predictions.append(prediction_data)

  with open(path + file_name, 'w') as file:
    json.dump(predictions, file, indent=4)


# Endpoint para realizar la predicción
@app.post("/predict")
def predict(data: InputData):
    # Crear DataFrame a partir de los datos de entrada
    user_data = pd.DataFrame([data.dict()])
    
    # Renombrar las columnas para que coincidan con lo que el modelo espera
    user_data.rename(columns={
        'Dominio': 'dominio',
        'Tecnologia':'Tec',
        'Avg_Session_Length': 'Avg. Session Length',
        'Time_on_App': 'Time on App',
        'Time_on_Website': 'Time on Website',
        'Length_of_Membership': 'Length of Membership'
    }, inplace=True)

    # Realizar predicción
    yhat = modelo.predict(user_data)

    # Guardar predicción con ID en el archivo JSON
    prediction_result = {
        "Email": user_data["Email"].values[0], 
        "Prediction": yhat[0]
    }
    
    save_prediction(prediction_result)

    return prediction_result

# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)