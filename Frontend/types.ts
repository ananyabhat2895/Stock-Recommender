
export type MarketCap = 'Small Cap' | 'Mid Cap' | 'Large Cap';
export type RiskTolerance = 'Low' | 'Medium' | 'High';
export type Asset = 'Equity' | 'Mutual Funds' | 'ETFs';
export type TimeHorizonUnit = 'Months' | 'Years' | 'Specific Date';

export interface FormData {
  initialAmount: number;
  timeHorizonValue: number;
  timeHorizonUnit: TimeHorizonUnit;
  specificDate: string;
  marketCap: MarketCap;
  riskTolerance: RiskTolerance;
  assets: Asset[];
}

export interface PortfolioDistribution {
  name: string;
  percentage: number;
}

export interface GeminiResponse {
  summary: string;
  portfolio: PortfolioDistribution[];
}
