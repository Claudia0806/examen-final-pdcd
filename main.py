from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from fastapi import Security, Depends, FastAPI, HTTPException
from fastapi.security.api_key import APIKeyQuery, APIKey
import pickle

from starlette.status import HTTP_403_FORBIDDEN
# from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Desarrollo de proyectos estudiantiles",
              description="Esta API es capaz de clasificar los proyectos en funded or  not funded",
              version="0.0.1")

API_KEY = "1234567asdfgh"
API_KEY_NAME = "access_token"

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_query: str = Security(api_key_query)):

    if api_key_query == API_KEY:
        return api_key_query
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )


class Proyect(BaseModel):
    description: str


@app.on_event("startup")
def load_model():

    global models
    global models2

    with open("LR.pickle", "rb") as openfile:
        models = pickle.load(openfile)
    with open("tfidf_model.pickle", "rb") as openfile:
        models2 = pickle.load(openfile)


@app.get("/api/v1/classify")
def classify_funded(proyect: Proyect, api_key: APIKey = Depends(get_api_key)):
    text = proyect.description
    # definimos pero de la predicion del modelo
    text2 = models2.transform([text]).toarray()
    pred = models.predict(text2)
    # ahora nos muestra la información
    # creamos un diccionario
    dict1= {0: "Funded",
            1: "No Funded"}

    return {"Proyect Education": dict1.get(pred[0]),
            "Desc": "Predicción hecha correctamente"}


@app.get("/")
def home():
    return{"Desc": "Health Check"}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
