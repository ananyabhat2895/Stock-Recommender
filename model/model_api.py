from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from model import run_model

app = FastAPI()

class RequestPayload(BaseModel):
    initialNetAmount: float
    timeHorizon: int
    marketCap: str
    riskTolerance: str
    assetPreferences: List[str]
    topN: int = 10

@app.post("/predict-stocks")
def predict_stocks(payload: RequestPayload):
    results = run_model(
        marketCap=payload.marketCap,
        riskTolerance=payload.riskTolerance,
        timeHorizon=payload.timeHorizon,
        assetPreferences=payload.assetPreferences,
        initialNetAmount=payload.initialNetAmount,
        top_n=payload.topN,
    )
    return {
        "recommendedStocks": results,
        "initialInvestment": payload.initialNetAmount,
        "timeHorizon": payload.timeHorizon,
    }
