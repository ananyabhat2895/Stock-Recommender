import React, { useState, useCallback, useMemo } from "react";

// Types
import type {
  FormData,
  MarketCap,
  RiskTolerance,
  Asset,
  TimeHorizonUnit,
} from "../../types";

// Constants
import {
  MARKET_CAP_OPTIONS,
  RISK_TOLERANCE_OPTIONS,
  ASSET_PREFERENCES,
  InfoIcon,
  ChevronDownIcon,
  XIcon,
} from "../../constants";

// Services
import { getMLRecommendations } from "../../services/mlService";

// Components
import Button from "../ui/Button";
import Card from "../ui/Card";
import Tooltip from "../ui/Tooltip";
import MLRecommendations from "./MLRecommendations";

const InsightForm: React.FC = () => {
  // ---------------------------
  // FORM STATE
  // ---------------------------
  const [formData, setFormData] = useState<FormData>({
    initialAmount: 10000,
    timeHorizonValue: 5,
    timeHorizonUnit: "Years",
    specificDate: "",
    marketCap: "Mid Cap",
    riskTolerance: "Medium",
    assets: ["ENERGY", "Basic Materials"],
  });

  // ML States
  const [mlLoading, setMlLoading] = useState(false);
  const [mlItems, setMlItems] = useState<
    { symbol: string; score: number }[] | null
  >(null);
  const [resultStocks, setResultStocks] = useState<any[]>([]);

  // UI State
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // ---------------------------
  // HELPERS
  // ---------------------------

  const handleInputChange = useCallback(
    <T extends keyof FormData>(field: T, value: FormData[T]) => {
      setFormData((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleAssetToggle = useCallback((asset: Asset) => {
    setFormData((prev) => {
      const newList = prev.assets.includes(asset)
        ? prev.assets.filter((a) => a !== asset)
        : [...prev.assets, asset];

      if (newList.length > 3) return prev;
      return { ...prev, assets: newList };
    });
  }, []);

  const mapMarketCapToBackend = (value: MarketCap) => {
    switch (value) {
      case "Small Cap":
        return "SMALL_CAP";
      case "Mid Cap":
        return "MID_CAP";
      case "Large Cap":
        return "LARGE_CAP";
      default:
        return "UNKNOWN";
    }
  };

  const computeTimeHorizonMonths = () => {
    if (formData.timeHorizonUnit === "Years")
      return formData.timeHorizonValue * 12;

    if (formData.timeHorizonUnit === "Months")
      return formData.timeHorizonValue;

    if (
      formData.timeHorizonUnit === "Specific Date" &&
      formData.specificDate
    ) {
      const today = new Date();
      const target = new Date(formData.specificDate);
      if (target <= today) return undefined;

      const diffMs = target.getTime() - today.getTime();
      return Math.round(diffMs / (1000 * 60 * 60 * 24 * 30));
    }

    return undefined;
  };

  const riskMeterStyle = useMemo(() => {
    const riskMap: Record<RiskTolerance, number> = {
      Low: 25,
      Medium: 60,
      High: 95,
    };
    const percentage = riskMap[formData.riskTolerance];

    return {
      width: `${percentage}%`,
      background: `linear-gradient(90deg, hsl(100, 70%, 50%), hsl(50, 80%, 50%) ${percentage}%, hsl(0, 80%, 50%) 100%)`,
    };
  }, [formData.riskTolerance]);

  // ---------------------------
  // SUBMIT HANDLER
  // ---------------------------

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (formData.initialAmount < 1000) {
      setError("Initial amount must be at least ₹1,000.");
      return;
    }

    const months = computeTimeHorizonMonths();
    if (!months || months <= 0) {
      setError("Please provide a valid time horizon.");
      return;
    }

    setError(null);
    setLoading(true);
    setMlItems(null);
    setResultStocks([]);
    setMlLoading(true);

    try {
      // Prepare ML payload
      const mlReq = {
        initialNetAmount: formData.initialAmount,
        timeHorizon: months,
        marketCap: mapMarketCapToBackend(formData.marketCap),
        riskTolerance: formData.riskTolerance,
        assetPreferences: formData.assets,
        topN: 10,
      };

      const mlResp = await getMLRecommendations(mlReq);

      // Small ML list
      setMlItems(
        mlResp.recommendedStocks?.map((s: any) => ({
          symbol: s.symbol,
          score: s.predicted_percent, // FIXED
        })) || []
      );

      // Full detailed cards
      setResultStocks(mlResp.recommendedStocks || []);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unexpected error occurred.");
    } finally {
      setLoading(false);
      setMlLoading(false);
    }
  };

  // ---------------------------
  // JSX RENDER
  // ---------------------------

  return (
    <section id="insight-form" className="py-16 md:py-24 animate-fadeIn">
      {/* HEADER */}
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold">
          Build Your Investment Profile
        </h2>
        <p className="mt-3 max-w-xl mx-auto text-foreground/70">
          Enter your details below to get stock recommendations.
        </p>
      </div>

      {/* FORM CARD */}
      <Card className="max-w-4xl mx-auto p-6 md:p-8">
        <form
          onSubmit={handleSubmit}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {/* Amount */}
          <div>
            <label className="flex items-center text-sm font-medium text-foreground/80 mb-2">
              Initial Net Amount (₹)
            </label>

            <div className="relative">
              <span className="absolute inset-y-0 left-3 flex items-center text-foreground/50">
                ₹
              </span>
              <input
                type="text"
                value={formData.initialAmount.toLocaleString("en-IN")}
                onChange={(e) => {
                  const value = parseInt(
                    e.target.value.replace(/,/g, ""),
                    10
                  );
                  handleInputChange(
                    "initialAmount",
                    isNaN(value) ? 0 : value
                  );
                }}
                className="w-full pl-7 pr-4 py-2 bg-background border border-foreground/20 rounded-md"
              />
            </div>
          </div>

          {/* TIME HORIZON */}
          <div>
            <label className="text-sm font-medium text-foreground/80 mb-2 block">
              Time Horizon
            </label>

            <div className="flex gap-2">
              <div className="relative flex-grow">
                <select
                  value={formData.timeHorizonValue}
                  onChange={(e) =>
                    handleInputChange(
                      "timeHorizonValue",
                      Number(e.target.value)
                    )
                  }
                  disabled={
                    formData.timeHorizonUnit === "Specific Date"
                  }
                  className="w-full pr-8 py-2 pl-3 bg-background border border-foreground/20 rounded-md"
                >
                  {Array.from({ length: 50 }, (_, i) => i + 1).map(
                    (v) => (
                      <option key={v}>{v}</option>
                    )
                  )}
                </select>
                <ChevronDownIcon className="absolute right-2 top-1/2 -translate-y-1/2 h-5 w-5 text-foreground/50" />
              </div>

              <div className="relative">
                <select
                  value={formData.timeHorizonUnit}
                  onChange={(e) =>
                    handleInputChange(
                      "timeHorizonUnit",
                      e.target.value as TimeHorizonUnit
                    )
                  }
                  className="w-full pr-8 py-2 pl-3 bg-background border border-foreground/20 rounded-md"
                >
                  <option>Days</option>
                  <option>Months</option>
                  <option>Years</option>
                </select>
              </div>
            </div>
          </div>

          {/* MARKET CAP */}
          <div>
            <label className="text-sm font-medium text-foreground/80 mb-2 block">
              Market Cap Preference
            </label>

            <div className="relative">
              <select
                value={formData.marketCap}
                onChange={(e) =>
                  handleInputChange(
                    "marketCap",
                    e.target.value as MarketCap
                  )
                }
                className="w-full pr-8 py-2 pl-3 bg-background border border-foreground/20 rounded-md"
              >
                {MARKET_CAP_OPTIONS.map((opt) => (
                  <option key={opt.value}>{opt.value}</option>
                ))}
              </select>
            </div>
          </div>

          {/* RISK TOLERANCE */}
          <div>
            <label className="flex items-center text-sm font-medium text-foreground/80 mb-2">
              Risk Tolerance
            </label>

            <div className="flex gap-2 mb-2">
              {RISK_TOLERANCE_OPTIONS.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() =>
                    handleInputChange("riskTolerance", opt)
                  }
                  className={`flex-1 py-2 rounded-md text-sm ${
                    formData.riskTolerance === opt
                      ? "bg-primary text-white"
                      : "bg-background border border-foreground/20"
                  }`}
                >
                  {opt}
                </button>
              ))}
            </div>

            <div className="w-full bg-foreground/10 rounded-full h-2.5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500 ease-out"
                style={riskMeterStyle}
              />
            </div>
          </div>

          {/* ASSETS */}
          <div className="md:col-span-2">
            <label className="text-sm font-medium text-foreground/80 mb-2 block">
              Sector / Asset Preferences (Max 3)
            </label>

            <div className="flex flex-wrap gap-2 mb-2">
              {formData.assets.map((asset) => (
                <span
                  key={asset}
                  className="flex items-center bg-primary/10 text-primary text-sm font-medium px-2 py-1 rounded-full"
                >
                  {asset}
                  <button
                    type="button"
                    onClick={() => handleAssetToggle(asset)}
                  >
                    <XIcon className="h-3 w-3 ml-1.5" />
                  </button>
                </span>
              ))}
            </div>

            <div className="flex flex-wrap gap-2">
              {ASSET_PREFERENCES.filter(
                (opt) => !formData.assets.includes(opt)
              ).map((opt) => (
                <button
                  key={opt}
                  type="button"
                  disabled={formData.assets.length >= 3}
                  onClick={() => handleAssetToggle(opt)}
                  className="px-3 py-1.5 text-sm rounded-md bg-background border border-foreground/20"
                >
                  + {opt}
                </button>
              ))}
            </div>
          </div>

          {/* SUBMIT */}
          <div className="md:col-span-2 text-center mt-4">
            <Button
              type="submit"
              size="lg"
              className="w-full md:w-auto"
              disabled={loading}
            >
              {loading ? "Generating..." : "Generate Insights"}
            </Button>
            {error && (
              <p className="text-red-500 text-sm mt-4">{error}</p>
            )}
          </div>
        </form>
      </Card>

      {/* LOADING */}
      {loading && (
        <div className="text-center mt-8">
          <div className="inline-block animate-spin h-8 w-8 rounded-full border-b-2 border-primary"></div>
          <p className="mt-2 text-foreground/70">
            Analyzing your profile...
          </p>
        </div>
      )}

      {/* --- Small ML Top List --- */}
      {(mlLoading || mlItems) && (
        <div className="mt-12 animate-fadeIn">
          <MLRecommendations loading={mlLoading} items={mlItems} />
        </div>
      )}

      {/* --- Detailed Stock Cards --- */}
      {resultStocks.length > 0 && (
        <div className="mt-14 max-w-4xl mx-auto animate-fadeIn">
          <h2 className="text-2xl font-bold mb-6">
            📊 Recommended Stocks (Detailed)
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {resultStocks.map((stock) => {
              const profit = stock.profit;
              const profitColor =
                profit >= 0 ? "text-green-600" : "text-red-600";

              return (
                <div
                  key={stock.symbol}
                  className="p-5 border border-foreground/20 rounded-xl bg-background shadow-sm"
                >
                  <p className="text-2xl font-semibold mb-1">
                    {stock.symbol}
                  </p>

                  <p className="text-sm text-foreground/70 mb-2">
                    Close Price:{" "}
                    <span className="font-medium">
                      ₹{stock.close}
                    </span>
                  </p>

                  <p className="text-sm text-foreground/70 mb-2">
                    Predicted Return:{" "}
                    <span className="font-medium">
                      {stock.predicted_percent.toFixed(2)}%
                    </span>
                  </p>

                  <p className="text-sm text-foreground/70 mb-2">
                    Final Value:{" "}
                    <span className="font-medium">
                      ₹{stock.final_amount.toLocaleString("en-IN")}
                    </span>
                  </p>

                  <p
                    className={`text-sm font-semibold ${profitColor}`}
                  >
                    Profit: ₹
                    {profit.toLocaleString("en-IN")}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
};

export default InsightForm;
