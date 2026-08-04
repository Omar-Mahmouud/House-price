import json
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="House Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved pipeline model and location options from backend folder
try:
    model = joblib.load("house_price.pkl")
    with open("locations.json", "r") as f:
        allowed_locations = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model files in backend: {e}")


class PropertyFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    area_sqft: float = Field(..., gt=0, json_schema_extra={"example": 1200.0})
    floor_num: int = Field(0, ge=-1, json_schema_extra={"example": 3})
    Bathroom: float = Field(1.0, ge=0, json_schema_extra={"example": 2.0})
    Balcony: float = Field(0.0, ge=0, json_schema_extra={"example": 1.0})
    Car_Parking: float = Field(
        0.0, ge=0, alias="Car Parking", json_schema_extra={"example": 1.0}
    )
    location_grouped: str = Field(
        "other", json_schema_extra={"example": "mumbai"}
    )
    society_grouped: str = Field("other", json_schema_extra={"example": "other"})
    Furnishing: str = Field(
        "Unfurnished", json_schema_extra={"example": "Semi-Furnished"}
    )
    Transaction: str = Field("Resale", json_schema_extra={"example": "Resale"})
    Ownership: str = Field("Freehold", json_schema_extra={"example": "Freehold"})
    facing: str = Field("North", json_schema_extra={"example": "East"})


@app.get("/")
def home():
    return {"status": "online", "message": "House Price Prediction API is running!"}


@app.get("/locations")
def get_locations():
    return {"locations": allowed_locations}


@app.post("/predict")
def predict_price(features: PropertyFeatures):
    try:
        input_data = features.model_dump(by_alias=True)
        df_input = pd.DataFrame([input_data])

        if df_input["location_grouped"].iloc[0] not in allowed_locations:
            df_input["location_grouped"] = "other"

        predicted_log = model.predict(df_input)[0]
        predicted_price_rupees = float(np.expm1(predicted_log))

        return {
            "predicted_price_rupees": round(predicted_price_rupees, 2),
            "predicted_price_formatted": f"₹{predicted_price_rupees:,.2f}",
            "log_price": round(float(predicted_log), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)