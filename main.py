import numpy as np
import pandas as pd 
from tensorflow.keras.models import load_model
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],  # or specify methods like ["GET", "POST"]
    allow_headers=["*"],  # or specify headers like ["Content-Type"]
)
class Input(BaseModel):
    id:int
# Load the .keras model
model = keras.models.load_model('model.keras')
# model = load_model('lstm_model.h5')  
df=pd.read_csv('synthetic_sensor_data.csv')
# model.summary()
sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
def create_sequence_from_custom_data(df, seq_length=30):
    """
    Given sensor data for one machine, extract the latest sequence.
    Returns a numpy array with shape (1, seq_length, num_features)
    """
    if len(df) < seq_length:
        raise ValueError("Not enough data to form a complete sequence.")
    # Use the last seq_length rows
    sequence = df[sensor_cols].values[-seq_length:]
    return np.expand_dims(sequence, axis=0) 

custom_data = df[df['machine_id'] == 1].copy()
X_custom = create_sequence_from_custom_data(custom_data, seq_length=30)
# print("Predicted Remaining Useful Life (RUL) for Machine 1:", predicted_rul[0][0])
@app.get("/")
async def home():
    return {"status": "OK", "message": "Working"}
@app.post("/predict")
async def pred(data:Input):
    # print(id)
    print(data.id)
    custom_data = df[df['machine_id'] == data.id].copy()
    X_custom = create_sequence_from_custom_data(custom_data, seq_length=30)
    predicted_rul = model.predict(X_custom)
    print(predicted_rul)
    return {"status": "OK", "message": "Working","answer":str(predicted_rul[0][0])}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)