import React from "react";
import Card from "../ui/Card";

interface MLItem {
  symbol: string;
  predicted_return: number;
  initialInvestment: number;
}

interface Props {
  loading: boolean;
  items: MLItem[] | null;
  onRefresh?: () => void;
}

const MLRecommendations: React.FC<Props> = ({ loading, items, onRefresh }) => {
  
  // DEBUG: Check what data looks like
  console.log("🔥 ML TOP ITEMS RECEIVED:", items);

  return (
    <Card className="p-6 mb-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold">ML — Top Predicted Stocks</h3>

        <button
          onClick={onRefresh}
          className="px-3 py-1 rounded-md bg-secondary text-white text-sm"
        >
          Refresh
        </button>
      </div>

      {loading && <p>Computing ML recommendations…</p>}

      {!loading && items && items.length > 0 && (
        <ul className="space-y-2">
          {items.map((it, idx) => {
            const finalValue = it.initialInvestment * (1 + it.predicted_return);

            return (
              <li
                key={it.symbol}
                className="flex justify-between items-center p-2 bg-background rounded-md"
              >
                <div className="flex items-center gap-3">
                  <span>{idx + 1}.</span>
                  <span className="font-semibold">{it.symbol}</span>
                </div>

                {/* <span>₹{finalValue.toLocaleString("en-IN")}</span> */}
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
};

export default MLRecommendations;
