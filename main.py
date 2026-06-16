from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

class Message(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# spam_model.pkl is a full sklearn Pipeline (preprocessor + TF-IDF + classifier)
model = joblib.load("spam_model.pkl")

@app.get("/")
def root():
    return FileResponse("spam-classifier-frontend/index.html")

@app.post("/predict")
def predict(msg: Message):
    pred = model.predict([msg.text])[0]
    result = "spam" if pred == 1 else "ham"
    return {"prediction": result}
