
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class Message(BaseModel):
    text: str

app = FastAPI()

# Load the model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.get("/")
def root():
    return {"status": "Spam Classifier Ready"}

@app.post("/predict")
def predict(msg: Message):
    text = [msg.text]
    vector = vectorizer.transform(text)
    pred = model.predict(vector)[0]
    result = "spam" if pred == 1 else "ham"
    return {"prediction": result}