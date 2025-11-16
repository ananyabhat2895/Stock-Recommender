// Backend/app.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

// Increase timeouts
app.use((req, res, next) => {
  req.setTimeout(3000000);
  res.setTimeout(3000000);
  next();
});

app.use(express.json());
app.use(cors({ origin: true, credentials: true }));

// ML Server URL
const MODEL_SERVER = process.env.MODEL_SERVER_URL || "http://localhost:8000";

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "Backend running" });
});

// =======================
// ML RECOMMENDER ROUTE
// =======================
app.post("/api/ml-recommend", async (req, res) => {
  console.log("BODY RECEIVED:", req.body);

  try {
    const payload = {
      initialNetAmount: req.body.initialNetAmount,
      timeHorizon: req.body.timeHorizon,
      marketCap: req.body.marketCap,
      riskTolerance: req.body.riskTolerance,
      assetPreferences: req.body.assetPreferences,
      topN: 10
    };

    // Call Python server
    const response = await axios.post(
      `${MODEL_SERVER}/predict-stocks`,
      payload,
      { timeout: 3000000 }
    );

    const modelData = response.data;
    const initialInvestment = modelData.initialInvestment;

    // ================================
    // MAP PYTHON OUTPUT → UI FORMAT
    // ================================

    const top = modelData.recommendedStocks.map(s => {
      // Python may return:
      // 1. pred_return_fwd (decimal VERSION)
      // 2. predicted_percent (percentage VERSION)

      let predicted_decimal = 0;

      if (s.pred_return_fwd !== undefined) {
        predicted_decimal = s.pred_return_fwd;        // e.g., 0.12
      } else if (s.predicted_percent !== undefined) {
        predicted_decimal = s.predicted_percent / 100; // convert 12 → 0.12
      }

      const finalValue = initialInvestment * (1 + predicted_decimal);
      const profit = finalValue - initialInvestment;

      return {
        symbol: s.symbol,
        final_amount: finalValue,
        profit: profit,
        predicted_decimal: predicted_decimal   // pass to UI
      };
    });

    res.json({
      top,
      recommendedStocks: modelData.recommendedStocks,
      initialInvestment,
      timeHorizon: modelData.timeHorizon
    });

  } catch (err) {
    console.error("ML API error:", err?.response?.data || err);
    res.status(500).json({
      error: "ML Server unreachable",
      details: err?.response?.data || err.message,
    });
  }
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log("Backend running on port", PORT);
});
