import React from "react";
import Card from "../ui/Card";

type MLItem = { symbol: string; score: number };

interface Props {
  loading: boolean;
  items: MLItem[] | null;
  onRefresh?: () => void;
}

const MLRecommendations: React.FC<Props> = ({ loading, items, onRefresh }) => {
  return (
    <Card className="p-6 mb-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold">ML — Top Predicted Stocks</h3>
        <div>
          <button
            onClick={onRefresh}
            className="px-3 py-1 rounded-md bg-secondary text-white text-sm"
          >
            Refresh
          </button>
        </div>
      </div>

      {loading && (
        <p className="text-foreground/70">Computing ML recommendations…</p>
      )}

      {!loading && (!items || items.length === 0) && (
        <p className="text-foreground/70">
          No ML recommendations available for the selected filters.
        </p>
      )}

      {!loading && items && items.length > 0 && (
        <ul className="space-y-2">
          {items.map((it, idx) => (
            <li
              key={it.symbol}
              className="flex justify-between items-center p-2 bg-background rounded-md"
            >
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium">{idx + 1}.</span>
                <span className="font-semibold">{it.symbol}</span>
              </div>
              <div className="text-sm text-foreground/70">
                {(it.score * 100).toFixed(2)}%{" "}
                {/* shown as percent of predicted forward return */}
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
};

export default MLRecommendations;
