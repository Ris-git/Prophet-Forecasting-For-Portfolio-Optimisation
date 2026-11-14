"""LLM-powered financial analytics assistant for portfolio explainability."""

from __future__ import annotations

import json
import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

try:
    from google import genai
    from google.genai import types
except ImportError:  # Allows the independent OpenRouter fallback to run.
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from src.settings import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/free"


def get_assistant_error_message(error: Exception) -> str:
    """Translate provider failures into a useful, safe message for dashboard users.

    Provider error bodies can contain quota identifiers, project numbers, and setup
    links.  They are helpful in server logs but should never be shown in the chat.
    """
    error_text = str(error).lower()

    if any(marker in error_text for marker in ("429", "resource_exhausted", "rate_limit", "quota")):
        return (
            "⚠️ **The AI assistant is temporarily busy.** The language service has "
            "reached its request limit. Please wait a minute and try again. If this "
            "continues, the app owner needs to enable or increase the Gemini API quota."
        )

    if any(marker in error_text for marker in ("401", "403", "unauthenticated", "permission_denied")):
        return (
            "⚠️ **The AI assistant is not configured correctly right now.** Please ask "
            "the app owner to check the Gemini API key and its permissions."
        )

    if any(marker in error_text for marker in ("timeout", "timed out", "deadline_exceeded")):
        return (
            "⚠️ **The AI assistant took too long to respond.** Please try your question "
            "again in a moment."
        )

    return (
        "⚠️ **The AI assistant is unavailable right now.** Please try again in a "
        "moment. If the problem continues, ask the app owner to check the service logs."
    )


def _portfolio_rows_from_context(portfolio_context: str) -> list[dict[str, float | str]]:
    """Extract the compact asset table created by ``build_portfolio_context``."""
    rows: list[dict[str, float | str]] = []
    pattern = re.compile(
        r"^\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([+-]?[\d.]+)%\s*\|\s*([+-]?[\d.]+)%\s*\|$"
    )
    for line in portfolio_context.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        ticker, price, predicted_return, weight = match.groups()
        rows.append(
            {
                "ticker": ticker.strip(),
                "price": float(price),
                "return": float(predicted_return),
                "weight": float(weight),
            }
        )
    return rows


def offline_portfolio_explainer(user_message: str, portfolio_context: str) -> str:
    """Give a useful, data-grounded reply when the live language service is down.

    This deliberately covers common dashboard questions without claiming to be a
    substitute for the live assistant.  It only uses values supplied in the current
    portfolio context.
    """
    rows = _portfolio_rows_from_context(portfolio_context)
    question = user_message.lower()
    mode_notice = (
        "*Live AI is temporarily unavailable, so this is an offline explanation "
        "based only on the portfolio currently shown on the dashboard.*\n\n"
    )

    if not rows:
        return (
            mode_notice
            +
            "I don’t have enough current portfolio data to analyse that yet. Once the "
            "dashboard has loaded allocations and forecasts, I can explain their risk, "
            "weights, and model assumptions."
        )

    top = rows[0]
    highest_return = max(rows, key=lambda row: float(row["return"]))
    lowest_return = min(rows, key=lambda row: float(row["return"]))
    top_three = rows[:3]
    top_three_text = ", ".join(
        f"{row['ticker']} ({float(row['weight']):.2f}%)" for row in top_three
    )

    if any(term in question for term in ("risk", "trade-off", "return")):
        return (
            mode_notice
            +
            f"**Risk–return view:** the portfolio places its largest allocation in "
            f"**{top['ticker']} ({float(top['weight']):.2f}%)**. The strongest model "
            f"forecast in this snapshot is **{highest_return['ticker']} "
            f"({float(highest_return['return']):.2f}% expected return)**.\n\n"
            "The optimiser does not simply put all capital into the highest-return "
            "asset: it balances expected return against volatility and how assets move "
            "together. A larger weight therefore signals a better risk-adjusted fit in "
            "this model, not a guarantee of performance."
        )

    if any(term in question for term in ("weight", "allocation", "why is", "why are")):
        return (
            mode_notice
            +
            f"**Allocation view:** the three largest positions are {top_three_text}.\n\n"
            "Portfolio weights are driven by the Prophet-based expected returns, each "
            "asset’s historical volatility, and the covariance between assets. An asset "
            "can receive a lower weight even with a strong forecast if it adds too much "
            "overlapping risk; conversely, a moderate-return asset can help diversify the "
            "whole portfolio."
        )

    if any(term in question for term in ("walkthrough", "dashboard", "what it shows")):
        return (
            mode_notice
            +
            f"This dashboard currently covers **{len(rows)} assets**. It shows each "
            "asset’s Prophet forecast, predicted return, and Markowitz portfolio weight. "
            f"The largest allocation is {top['ticker']} at {float(top['weight']):.2f}%, "
            f"while {highest_return['ticker']} has the highest displayed expected return "
            f"at {float(highest_return['return']):.2f}%.\n\n"
            "Read the forecasts as model estimates, then use the weights to see how the "
            "optimiser balances those estimates against portfolio risk."
        )

    if any(term in question for term in ("markowitz", "optimis", "optimiz")):
        return (
            mode_notice
            +
            "**How the optimisation works:** it begins with the forecast-derived expected "
            "returns, estimates risk from historical return covariance, and selects weights "
            "that balance return and risk under the dashboard’s constraints.\n\n"
            f"In this snapshot, {top['ticker']} is the largest selected position "
            f"({float(top['weight']):.2f}%). That outcome reflects the full portfolio "
            "combination—not a standalone prediction that this asset will outperform."
        )

    if any(term in question for term in ("top", "compare", "contribution")):
        return (
            mode_notice
            +
            f"**Largest current allocations:** {top_three_text}. The highest displayed "
            f"expected return is {highest_return['ticker']} ({float(highest_return['return']):.2f}%), "
            f"and the lowest is {lowest_return['ticker']} ({float(lowest_return['return']):.2f}%).\n\n"
            "A direct contribution comparison needs each asset’s weight, return estimate, "
            "and covariance with the rest of the portfolio; the optimiser uses all three."
        )

    return (
        mode_notice
        +
        f"That’s a thoughtful question. The current snapshot shows {len(rows)} assets, "
        f"led by {top['ticker']} at {float(top['weight']):.2f}% allocation. I can’t "
        "verify claims beyond these forecasts and allocations without the live assistant, "
        "but I can still help you explore the model’s evidence.\n\n"
        "Try asking how the portfolio weights are determined, what the risk–return trade-off "
        "means, how Markowitz optimisation works, or how the largest allocations compare."
    )

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


def get_openrouter_api_key() -> str | None:
    """Get an OpenRouter key from deployment secrets or ignored local config."""
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        try:
            import streamlit as st

            if hasattr(st, "secrets"):
                if "OPENROUTER_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENROUTER_API_KEY"]
                elif (
                    "general" in st.secrets
                    and "OPENROUTER_API_KEY" in st.secrets["general"]
                ):
                    api_key = st.secrets["general"]["OPENROUTER_API_KEY"]
        except ImportError:
            pass

    if not api_key:
        local_config = Path(__file__).resolve().parent.parent / ".streamlit" / "openrouter.toml"
        try:
            with local_config.open("rb") as config_file:
                api_key = tomllib.load(config_file).get("OPENROUTER_API_KEY")
        except (FileNotFoundError, tomllib.TOMLDecodeError):
            pass

    return api_key


def chat_with_openrouter(
    user_message: str,
    portfolio_context: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str | None:
    """Request a free OpenRouter model as a backup for Gemini."""
    api_key = get_openrouter_api_key()
    if not api_key:
        return None

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        history_to_use = chat_history
        if chat_history[-1]["role"] == "user" and chat_history[-1]["content"] == user_message:
            history_to_use = chat_history[:-1]
        messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in history_to_use[-6:]
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "Here is the current portfolio data you should use to answer questions:\n\n"
                f"{portfolio_context}\n\n---\n\n"
                f"Now answer the following question based on this data:\n{user_message}"
            ),
        }
    )

    payload = json.dumps(
        {"model": OPENROUTER_MODEL, "messages": messages, "temperature": LLM_TEMPERATURE}
    ).encode("utf-8")
    request = Request(
        OPENROUTER_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Portfolio Forecast Dashboard",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed provider endpoint
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        logger.exception("Error calling OpenRouter API")
        return None


def get_llm_client() -> genai.Client | None:
    """
    Initialize and return Gemini client (new google-genai SDK).

    Returns:
        Configured Client if API key available, None otherwise
    """
    if genai is None:
        logger.warning("google-genai SDK is not installed; Gemini is unavailable")
        return None

    api_key = get_gemini_api_key()
    if not api_key:
        logger.warning("Gemini API key not found in environment or Streamlit secrets")
        return None

    return genai.Client(api_key=api_key)


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
    def backup_response() -> str:
        openrouter_response = chat_with_openrouter(user_message, portfolio_context, chat_history)
        if openrouter_response:
            return openrouter_response
        return offline_portfolio_explainer(user_message, portfolio_context)

    client = get_llm_client()
    if client is None:
        logger.warning("Gemini client unavailable; trying OpenRouter backup.")
        return backup_response()

    # Build the full message with portfolio context injected
    context_message = f"""Here is the current portfolio data you should use to answer questions:

{portfolio_context}

---

Now answer the following question based on this data:
{user_message}"""

    # Build conversation history in the new SDK format
    contents: list[types.Content] = []

    if chat_history:
        history_to_use = chat_history
        # Avoid duplicating the current user message if it's already in history
        if chat_history[-1]["content"] == user_message and chat_history[-1]["role"] == "user":
            history_to_use = chat_history[:-1]

        for msg in history_to_use[-6:]:  # Keep last 6 messages for context
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    # Append the current user message
    contents.append(types.Content(role="user", parts=[types.Part(text=context_message)]))

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )
        text = response.text
        if not text or not text.strip():
            logger.warning("Gemini returned an empty response.")
            return backup_response()

        # Guard: detect mid-markdown truncation (odd number of ** markers → unclosed bold)
        if text.count("**") % 2 != 0:
            logger.warning("LLM response appears truncated (unclosed markdown bold markers).")
            text += "\n\n> ⚠️ *Response was truncated. Try asking a more specific question.*"

        return text
    except Exception as error:
        # Keep the detailed exception in server logs for diagnosis, but do not leak
        # provider/project details into the customer-facing chat.
        logger.exception("Error calling Gemini API")
        logger.warning(get_assistant_error_message(error))
        return backup_response()


# Suggested questions for users
SUGGESTED_QUESTIONS = [
    "Give me a walkthrough of this dashboard and what it shows.",
    "Explain the risk-return trade-off in this portfolio.",
    "What factors influence the portfolio weights?",
    "How does the Markowitz optimization work?",
    "Compare the top 3 assets by their contribution to the portfolio.",
]
