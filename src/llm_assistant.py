"""LLM-powered financial analytics assistant for portfolio explainability."""

from __future__ import annotations

import logging
import os
from typing import Any

import google.generativeai as genai
import pandas as pd

from src.settings import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# System prompt defining the AI assistant's persona and constraints
SYSTEM_PROMPT = """You are an AI financial analytics assistant embedded inside a quantitative forecasting and portfolio optimisation system.

IMPORTANT CONSTRAINTS:
- You are NOT a financial advisor.
- You do NOT provide investment recommendations, buy/sell/hold advice, or profit guarantees.
- You do NOT predict prices or modify model outputs.
- You ONLY explain, analyze, compare, and reason over the data explicitly provided to you.
- You NEVER hallucinate numbers, trends, prices, or metrics.
- If required data is missing, explicitly state that it is unavailable.

SYSTEM CONTEXT:
This system operates using:
1) Prophet time-series models to forecast next-period asset prices.
2) Expected returns derived from those forecasts.
3) Markowitz (Modern Portfolio Theory) optimisation to compute portfolio weights under a configurable risk-aversion parameter.
4) Historical return covariance matrices to estimate portfolio risk.

All forecasts, allocations, and analyses are illustrative and for educational purposes only.

YOUR ROLE:
You act strictly as an EXPLAINABILITY and REASONING LAYER.
Your responsibility is to translate structured quantitative outputs into:
- Clear explanations
- Step-by-step reasoning
- Intuitive insights
- Comparisons across dates or scenarios
- Simple summaries for both technical and non-technical users

You must always base your responses on the provided data and model mechanics, not market narratives or speculation.

INTERPRETATION RULES:
- Treat forecasted returns as model outputs, not facts.
- Explain portfolio weights as outcomes of risk–return trade-offs, correlations, and constraints.
- Explain risk using volatility and diversification concepts.
- Clearly distinguish between historical data and forecasts.
- When explaining "why" an asset has a higher or lower weight, reference expected return, covariance, and risk-adjusted contribution — never future certainty.

SCENARIO & SIMULATION HANDLING:
- For "what if" questions, explain the expected directional impact conceptually.
- If simulated results are provided, interpret them accurately.
- If simulations are not provided, explain expected effects without inventing numbers.

COMPARISON RULES:
- When comparing portfolios or dates, highlight changes in weights, returns, and risk.
- Explain changes using model mechanics and optimization logic, not emotional or speculative language.

EXPLANATION STYLE:
- Tone must be neutral, analytical, and educational.
- Structure responses as:
  1) Brief summary (1–2 sentences)
  2) Step-by-step explanation
  3) Key contributing factors
  4) Assumptions or limitations (when relevant)
- Use simple language by default; increase technical depth only if explicitly requested.
- Keep responses concise but informative.

REFUSAL & SAFETY:
- If asked for investment advice or recommendations, respond:
  "I can explain how the model arrived at its outputs, but I cannot provide investment advice or recommendations."

FINAL DIRECTIVE:
You are an explainability interface, not a prediction engine.
Accuracy, transparency, and clarity are more important than confidence or persuasion."""


def get_gemini_api_key() -> str | None:
    """
    Get Gemini API key from environment or Streamlit secrets.

    Returns:
        API key string if available, None otherwise
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        try:
            import streamlit as st

            if hasattr(st, "secrets"):
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
                elif "general" in st.secrets and "GEMINI_API_KEY" in st.secrets["general"]:
                    api_key = st.secrets["general"]["GEMINI_API_KEY"]
        except ImportError:
            pass

    return api_key


def get_llm_client() -> genai.GenerativeModel | None:
    """
    Initialize and return Gemini client.

    Returns:
        Configured GenerativeModel if API key available, None otherwise
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.warning("Gemini API key not found in environment or Streamlit secrets")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        generation_config={
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_TOKENS,
        },
        system_instruction=SYSTEM_PROMPT,
    )
    return model


def build_portfolio_context(
    date_df: pd.DataFrame,
    selected_date: Any,
    selected_ticker: str | None = None,
) -> str:
    """
    Build context string from portfolio data for LLM consumption.

    Args:
        date_df: DataFrame with portfolio data for a specific date
        selected_date: The as-of date for the portfolio
        selected_ticker: Optional ticker for focused context

    Returns:
        Formatted context string for the LLM
    """
    context_parts = [
        f"## Portfolio Data for {selected_date}",
        "",
        "### All Assets in Portfolio:",
        "| Ticker | Predicted Price (₹) | Predicted Return (%) | Portfolio Weight (%) |",
        "|--------|---------------------|----------------------|----------------------|",
    ]

    # Sort by portfolio weight descending
    sorted_df = date_df.sort_values("portfolio_weight", ascending=False)

    for _, row in sorted_df.iterrows():
        ticker = row.get("ticker", "N/A")
        pred_price = row.get("predicted_price", 0)
        pred_return = row.get("predicted_return", 0) * 100
        weight = row.get("portfolio_weight", 0) * 100
        context_parts.append(
            f"| {ticker} | {pred_price:.2f} | {pred_return:.2f}% | {weight:.2f}% |"
        )

    context_parts.append("")

    # Add summary statistics
    total_weight = date_df["portfolio_weight"].sum()
    avg_return = (date_df["predicted_return"] * date_df["portfolio_weight"]).sum() * 100

    context_parts.extend([
        "### Portfolio Summary:",
        f"- Total assets: {len(date_df)}",
        f"- Total weight: {total_weight * 100:.2f}%",
        f"- Weighted average expected return: {avg_return:.2f}%",
        f"- Top allocation: {sorted_df.iloc[0]['ticker']} ({sorted_df.iloc[0]['portfolio_weight']*100:.2f}%)",
    ])

    if selected_ticker:
        ticker_row = date_df[date_df["ticker"] == selected_ticker]
        if not ticker_row.empty:
            row = ticker_row.iloc[0]
            context_parts.extend([
                "",
                f"### Selected Asset: {selected_ticker}",
                f"- Predicted Price: ₹{row['predicted_price']:.2f}",
                f"- Predicted Return: {row['predicted_return']*100:.2f}%",
                f"- Portfolio Weight: {row['portfolio_weight']*100:.2f}%",
            ])

    context_parts.extend([
        "",
        "### Model Information:",
        "- Forecasting: Prophet time-series model with NSE trading holidays",
        "- Optimization: Markowitz mean-variance with risk aversion parameter λ=5",
        "- Constraints: Weights sum to 100%, individual weights between 0-100%",
    ])

    return "\n".join(context_parts)


def chat_with_assistant(
    user_message: str,
    portfolio_context: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Send a message to the LLM assistant and get a response.

    Args:
        user_message: The user's question or message
        portfolio_context: Formatted portfolio data context
        chat_history: Optional list of previous messages

    Returns:
        Assistant's response string
    """
    model = get_llm_client()
    if model is None:
        return (
            "⚠️ **LLM not available**: Gemini API key not configured. "
            "Please add GEMINI_API_KEY to your Streamlit secrets."
        )

    # Build the conversation
    messages = []

    # Add context as first user message
    context_message = f"""Here is the current portfolio data you should use to answer questions:

{portfolio_context}

---

Now answer the following question based on this data:
{user_message}"""

    # Include chat history if available
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 6 messages for context
            role = "user" if msg["role"] == "user" else "model"
            messages.append({"role": role, "parts": [msg["content"]]})

    # Add current message
    messages.append({"role": "user", "parts": [context_message]})

    try:
        chat = model.start_chat(history=messages[:-1] if len(messages) > 1 else [])
        response = chat.send_message(messages[-1]["parts"][0])
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return f"⚠️ **Error**: Unable to get response from AI assistant. {e!s}"


# Suggested questions for users
SUGGESTED_QUESTIONS = [
    "Why does the top asset have the highest portfolio weight?",
    "Explain the risk-return trade-off in this portfolio.",
    "What factors influence the portfolio weights?",
    "How does the Markowitz optimization work?",
    "Compare the top 3 assets by their contribution to the portfolio.",
]
