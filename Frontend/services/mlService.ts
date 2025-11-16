// Frontend/services/mlService.ts
import axios from 'axios';

export type MLRequest = {
  marketCap?: string;
  riskTolerance?: string;
  timeHorizonMonths?: number;
  topN?: number;
};

export type MLResponse = {
  top: { symbol: string; score: number }[];
  count: number;
};

const backendUrl = process.env.REACT_APP_BACKEND_URL || ''; // if empty, relative path used

export async function getMLRecommendations(payload: MLRequest): Promise<MLResponse> {
  const url = backendUrl ? `${backendUrl}/api/ml-recommend` : `/api/ml-recommend`;
  const resp = await axios.post(url, payload, { timeout: 30000 });
  return resp.data;
}
