"""Tests for LLM assistant module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.llm_assistant import (
    SUGGESTED_QUESTIONS,
    SYSTEM_PROMPT,
    build_portfolio_context,
    chat_with_assistant,
    get_gemini_api_key,
    get_llm_client,
)


class TestSystemPrompt:
    """Test system prompt configuration."""

    def test_system_prompt_contains_constraints(self) -> None:
        """Test that system prompt contains key safety constraints."""
        assert "NOT a financial advisor" in SYSTEM_PROMPT
        assert "NOT provide investment recommendations" in SYSTEM_PROMPT
        assert "NEVER hallucinate" in SYSTEM_PROMPT

    def test_system_prompt_contains_context(self) -> None:
        """Test that system prompt describes the system context."""
        assert "Prophet" in SYSTEM_PROMPT
        assert "Markowitz" in SYSTEM_PROMPT
        assert "covariance" in SYSTEM_PROMPT

    def test_system_prompt_contains_refusal(self) -> None:
        """Test that system prompt contains refusal instructions."""
        assert "cannot provide investment advice" in SYSTEM_PROMPT


class TestSuggestedQuestions:
    """Test suggested questions configuration."""

    def test_suggested_questions_not_empty(self) -> None:
        """Test that suggested questions list is not empty."""
        assert len(SUGGESTED_QUESTIONS) > 0

    def test_suggested_questions_are_strings(self) -> None:
        """Test that all suggested questions are strings."""
        for question in SUGGESTED_QUESTIONS:
            assert isinstance(question, str)
            assert len(question) > 10  # Meaningful question length


class TestGetGeminiApiKey:
    """Test API key retrieval."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    def test_get_api_key_from_environment(self) -> None:
        """Test getting API key from environment variable."""
        api_key = get_gemini_api_key()
        assert api_key == "test-api-key"

    @patch.dict("os.environ", {}, clear=True)
    @patch.dict("sys.modules", {"streamlit": MagicMock(secrets={})})
    def test_get_api_key_returns_none_when_missing(self) -> None:
        """Test that None is returned when API key is not available."""
        api_key = get_gemini_api_key()
        assert api_key is None


class TestGetLlmClient:
    """Test LLM client creation."""

    @patch.dict("os.environ", {}, clear=True)
    @patch.dict("sys.modules", {"streamlit": MagicMock(secrets={})})
    def test_get_llm_client_without_api_key(self) -> None:
        """Test that None is returned when API key is not available."""
        client = get_llm_client()
        assert client is None

    @patch("src.llm_assistant.genai")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"})
    def test_get_llm_client_with_api_key(self, mock_genai: MagicMock) -> None:
        """Test that client is created when API key is available."""
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        client = get_llm_client()

        mock_genai.configure.assert_called_once_with(api_key="test-api-key")
        assert client is not None


class TestBuildPortfolioContext:
    """Test portfolio context building."""

    @pytest.fixture
    def sample_date_df(self) -> pd.DataFrame:
        """Create sample portfolio DataFrame."""
        return pd.DataFrame({
            "ticker": ["RELIANCE.NS", "INFY.NS", "TCS.NS"],
            "predicted_price": [2500.00, 1800.50, 4200.25],
            "predicted_return": [0.025, 0.015, 0.02],
            "portfolio_weight": [0.4, 0.35, 0.25],
        })

    def test_build_context_contains_tickers(self, sample_date_df: pd.DataFrame) -> None:
        """Test that context contains all ticker symbols."""
        context = build_portfolio_context(
            date_df=sample_date_df,
            selected_date=date(2024, 1, 31),
        )

        assert "RELIANCE.NS" in context
        assert "INFY.NS" in context
        assert "TCS.NS" in context

    def test_build_context_contains_prices(self, sample_date_df: pd.DataFrame) -> None:
        """Test that context contains predicted prices."""
        context = build_portfolio_context(
            date_df=sample_date_df,
            selected_date=date(2024, 1, 31),
        )

        assert "2500.00" in context
        assert "1800.50" in context

    def test_build_context_contains_selected_ticker(self, sample_date_df: pd.DataFrame) -> None:
        """Test that context highlights selected ticker."""
        context = build_portfolio_context(
            date_df=sample_date_df,
            selected_date=date(2024, 1, 31),
            selected_ticker="RELIANCE.NS",
        )

        assert "Selected Asset: RELIANCE.NS" in context

    def test_build_context_contains_model_info(self, sample_date_df: pd.DataFrame) -> None:
        """Test that context contains model information."""
        context = build_portfolio_context(
            date_df=sample_date_df,
            selected_date=date(2024, 1, 31),
        )

        assert "Prophet" in context
        assert "Markowitz" in context
        assert "risk aversion" in context


class TestChatWithAssistant:
    """Test chat functionality."""

    def test_chat_without_api_key(self) -> None:
        """Test that chat returns error message when API key is missing."""
        with patch("src.llm_assistant.get_llm_client", return_value=None):
            response = chat_with_assistant(
                user_message="Why is RELIANCE weighted highest?",
                portfolio_context="Test context",
            )

            assert "LLM not available" in response
            assert "GEMINI_API_KEY" in response

    @patch("src.llm_assistant.get_llm_client")
    def test_chat_returns_response(self, mock_get_client: MagicMock) -> None:
        """Test that chat returns LLM response."""
        # Setup mock
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "RELIANCE has the highest weight due to its high expected return."

        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response
        mock_get_client.return_value = mock_model

        response = chat_with_assistant(
            user_message="Why is RELIANCE weighted highest?",
            portfolio_context="Test context with RELIANCE data",
        )

        assert "RELIANCE" in response
        assert "highest weight" in response

    @patch("src.llm_assistant.get_llm_client")
    def test_chat_handles_api_error(self, mock_get_client: MagicMock) -> None:
        """Test that chat handles API errors gracefully."""
        mock_model = MagicMock()
        mock_model.start_chat.side_effect = Exception("API rate limit exceeded")
        mock_get_client.return_value = mock_model

        response = chat_with_assistant(
            user_message="Test question",
            portfolio_context="Test context",
        )

        assert "Error" in response
        assert "API rate limit exceeded" in response

    @patch("src.llm_assistant.get_llm_client")
    def test_chat_with_history(self, mock_get_client: MagicMock) -> None:
        """Test that chat includes conversation history."""
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Following up on the previous answer..."

        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response
        mock_get_client.return_value = mock_model

        chat_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        response = chat_with_assistant(
            user_message="Follow-up question",
            portfolio_context="Test context",
            chat_history=chat_history,
        )

        assert response is not None
        mock_model.start_chat.assert_called_once()
