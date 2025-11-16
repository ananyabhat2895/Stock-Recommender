// Frontend/services/mlService.ts
import axios from "axios";

export type MLRequest = {
  initialNetAmount: number;
  timeHorizon: number;
  marketCap: string;
  riskTolerance: string;
  assetPreferences: string[];
  topN?: number;
};


export type MLResponse = {
  top: { 
    symbol: string; 
    final_amount: number;
    profit: number;
  }[];
  recommendedStocks: any[];
};



const backendUrl = "http://localhost:5000";

export async function getMLRecommendations(payload: MLRequest): Promise<MLResponse> {
  const url = backendUrl
    ? `${backendUrl}/api/ml-recommend`
    : `/api/ml-recommend`;

  console.log("Sending payload:", payload);

  const resp = await axios.post(url, payload, { timeout: 0 });
  return resp.data;
}
