
import { GoogleGenAI, Type } from "@google/genai";
import type { FormData, GeminiResponse } from '../types';

export async function generateInsights(formData: FormData): Promise<GeminiResponse> {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

  const {
    initialAmount,
    timeHorizonValue,
    timeHorizonUnit,
    specificDate,
    marketCap,
    riskTolerance,
    assets
  } = formData;

  const timeHorizonText = timeHorizonUnit === 'Specific Date'
    ? `until ${specificDate}`
    : `${timeHorizonValue} ${timeHorizonUnit}`;

  const prompt = `
    Based on the following investor profile, provide a recommended stock portfolio distribution and a brief, encouraging summary sentence for the user.

    Investor Profile:
    - Initial Investment: ${initialAmount.toLocaleString('en-IN', { style: 'currency', currency: 'INR' })}
    - Time Horizon: ${timeHorizonText}
    - Market Cap Preference: ${marketCap}
    - Risk Tolerance: ${riskTolerance}
    - Preferred Asset Classes: ${assets.join(', ')}

    Return a JSON object. The 'summary' should be a single, encouraging sentence. The 'portfolio' should be an array of objects, each with a 'name' (e.g., 'Large Cap Equity', 'Mid Cap Mutual Fund') and a 'percentage' value. The percentages must sum to 100.
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: {
              type: Type.STRING,
              description: 'A single, encouraging summary sentence for the user.'
            },
            portfolio: {
              type: Type.ARRAY,
              description: 'An array of portfolio distribution objects. Percentages must sum to 100.',
              items: {
                type: Type.OBJECT,
                properties: {
                  name: {
                    type: Type.STRING,
                    description: 'The name of the asset category.'
                  },
                  percentage: {
                    type: Type.NUMBER,
                    description: 'The percentage allocation for this category.'
                  }
                },
                required: ['name', 'percentage'],
              }
            }
          },
          required: ['summary', 'portfolio'],
        },
      },
    });

    const jsonString = response.text.trim();
    const result = JSON.parse(jsonString);
    
    // Basic validation
    if (!result.summary || !Array.isArray(result.portfolio)) {
        throw new Error("Invalid response structure from API.");
    }

    return result as GeminiResponse;

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw new Error("Failed to generate insights. Please check your API key and try again.");
  }
}
