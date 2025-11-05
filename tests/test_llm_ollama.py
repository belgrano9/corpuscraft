"""Tests for Ollama LLM backend."""

import json
from unittest.mock import MagicMock, patch

import pytest

from corpuscraft.llm.ollama import OllamaLLM


class TestOllamaLLM:
    """Tests for OllamaLLM class."""

    @patch("corpuscraft.llm.ollama.Client")
    def test_initialization(self, mock_client_class: MagicMock) -> None:
        """Test OllamaLLM initialization."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(
            model="llama3.1:8b",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=2048,
        )

        assert llm.model == "llama3.1:8b"
        assert llm.base_url == "http://localhost:11434"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048
        mock_client_class.assert_called_once_with(host="http://localhost:11434")

    @patch("corpuscraft.llm.ollama.Client")
    def test_model_verification_success(self, mock_client_class: MagicMock) -> None:
        """Test successful model verification."""
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "mistral"},
            ]
        }
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")

        # Should not raise any warnings or errors
        assert llm.model == "llama3.1:8b"
        mock_client.list.assert_called_once()

    @patch("corpuscraft.llm.ollama.Client")
    def test_model_verification_warning(self, mock_client_class: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
        """Test model verification with missing model."""
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [{"name": "mistral"}]
        }
        mock_client_class.return_value = mock_client

        with caplog.at_level("WARNING"):
            llm = OllamaLLM(model="nonexistent-model")

        assert llm.model == "nonexistent-model"
        assert "not found locally" in caplog.text

    @patch("corpuscraft.llm.ollama.Client")
    def test_model_verification_connection_error(
        self, mock_client_class: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test model verification with connection error."""
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_client_class.return_value = mock_client

        with caplog.at_level("ERROR"):
            llm = OllamaLLM(model="llama3.1:8b")

        assert "Failed to connect to Ollama server" in caplog.text
        assert "Make sure Ollama is running" in caplog.text

    @patch("corpuscraft.llm.ollama.Client")
    def test_generate(self, mock_client_class: MagicMock) -> None:
        """Test text generation."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.generate.return_value = {
            "response": "This is a generated response."
        }
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")
        response = llm.generate("Test prompt")

        assert response == "This is a generated response."
        mock_client.generate.assert_called_once_with(
            model="llama3.1:8b",
            prompt="Test prompt",
            options={
                "temperature": 0.7,
                "num_predict": 2048,
            },
        )

    @patch("corpuscraft.llm.ollama.Client")
    def test_generate_with_custom_params(self, mock_client_class: MagicMock) -> None:
        """Test text generation with custom parameters."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.generate.return_value = {
            "response": "Custom response."
        }
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b", temperature=0.5, max_tokens=1024)
        response = llm.generate("Test prompt", temperature=0.9, max_tokens=512)

        assert response == "Custom response."
        mock_client.generate.assert_called_once_with(
            model="llama3.1:8b",
            prompt="Test prompt",
            options={
                "temperature": 0.9,  # Override
                "num_predict": 512,  # Override
            },
        )

    @patch("corpuscraft.llm.ollama.Client")
    def test_generate_error(self, mock_client_class: MagicMock) -> None:
        """Test error handling in generation."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.generate.side_effect = Exception("Generation error")
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")

        with pytest.raises(Exception, match="Generation error"):
            llm.generate("Test prompt")

    @patch("corpuscraft.llm.ollama.Client")
    def test_batch_generate(self, mock_client_class: MagicMock) -> None:
        """Test batch text generation."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.generate.side_effect = [
            {"response": "Response 1"},
            {"response": "Response 2"},
            {"response": "Response 3"},
        ]
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = llm.batch_generate(prompts)

        assert len(responses) == 3
        assert responses[0] == "Response 1"
        assert responses[1] == "Response 2"
        assert responses[2] == "Response 3"
        assert mock_client.generate.call_count == 3

    @patch("corpuscraft.llm.ollama.Client")
    def test_batch_generate_with_errors(
        self, mock_client_class: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test batch generation with some errors."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.generate.side_effect = [
            {"response": "Response 1"},
            Exception("Error in generation"),
            {"response": "Response 3"},
        ]
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with caplog.at_level("ERROR"):
            responses = llm.batch_generate(prompts)

        assert len(responses) == 3
        assert responses[0] == "Response 1"
        assert responses[1] == ""  # Error should result in empty string
        assert responses[2] == "Response 3"
        assert "Error in batch generation" in caplog.text

    @patch("corpuscraft.llm.ollama.Client")
    def test_chat(self, mock_client_class: MagicMock) -> None:
        """Test chat completion."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.chat.return_value = {
            "message": {"content": "Chat response"}
        }
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = llm.chat(messages)

        assert response == "Chat response"
        mock_client.chat.assert_called_once_with(
            model="llama3.1:8b",
            messages=messages,
            options={
                "temperature": 0.7,
                "num_predict": 2048,
            },
        )

    @patch("corpuscraft.llm.ollama.Client")
    def test_chat_with_custom_params(self, mock_client_class: MagicMock) -> None:
        """Test chat completion with custom parameters."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.chat.return_value = {
            "message": {"content": "Chat response"}
        }
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")
        messages = [{"role": "user", "content": "Hello"}]
        response = llm.chat(messages, temperature=0.3, max_tokens=100)

        assert response == "Chat response"
        mock_client.chat.assert_called_once_with(
            model="llama3.1:8b",
            messages=messages,
            options={
                "temperature": 0.3,
                "num_predict": 100,
            },
        )

    @patch("corpuscraft.llm.ollama.Client")
    def test_chat_error(self, mock_client_class: MagicMock) -> None:
        """Test error handling in chat."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client.chat.side_effect = Exception("Chat error")
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(Exception, match="Chat error"):
            llm.chat(messages)

    @patch("corpuscraft.llm.ollama.Client")
    def test_repr(self, mock_client_class: MagicMock) -> None:
        """Test string representation."""
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_client_class.return_value = mock_client

        llm = OllamaLLM(model="llama3.1:8b")
        assert repr(llm) == "OllamaLLM(model=llama3.1:8b)"
