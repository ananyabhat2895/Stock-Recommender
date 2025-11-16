// Backend/app.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

app.use(express.json());
app.use(cors({ origin: true, credentials: true }));

// ENV
const MODEL_SERVER = process.env.MODEL_SERVER_URL || "http://localhost:8000";

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "Backend running" });
});

// ML Recommendation route
app.post("/api/ml-recommend", async (req, res) => {
  try {
    const payload = {
      marketCap: req.body.marketCap,
      riskTolerance: req.body.riskTolerance,
      timeHorizonMonths: req.body.timeHorizonMonths,
      topN: 10,
    };

    const response = await axios.post(
      `${MODEL_SERVER}/predict-stocks`,
      payload
    );
    res.json(response.data);
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
