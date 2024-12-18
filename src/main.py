import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from inference import get_prediction, load_model

class Prediction(BaseModel):
    class_idx: int
    class_name: str
    confidence_score: float

app = FastAPI()

# Load environment variables from the .env file
load_dotenv("../.env")
# Access the environment variables
MODEL_PATH = os.getenv('MODEL_PATH')
device = os.getenv('DEVICE', 'cpu')
model = load_model(MODEL_PATH).to(device)

@app.post("/predict", response_model=Prediction)
async def predict(request: str):
    """
    Predict an HTTP request log by analyzing its attributes to detect anomalies.

    Args: request (str): The HTTP request log to classify.

    Returns: Prediction: The predicted class, its index and confidence score (probability).

    **EXAMPLE**

    INPUT: 

    GET /openautoclassifieds/friendmail.php?listing=<script>alert(document.domain);</script> HTTP/1.1
    User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
    Pragma: no-cache
    Cache-control: no-cache
    Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
    Accept-Encoding: x-gzip, x-deflate, gzip, deflate
    Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
    Accept-Language: en
    Host: localhost:8080
    Connection: close

    OUTPUT:

    {
    "class_idx": 0,
    "class_name": "anomaly",
    "confidence_score": 0.6069717407226562
    }
    """
    prediction = get_prediction(model, request, device)
    return prediction