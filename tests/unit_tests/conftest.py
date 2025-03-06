"""Shared fixtures for OpenGradient unit tests."""

import pytest


@pytest.fixture
def mock_env(monkeypatch) -> None:
    """Set up mock environment variables for OpenGradient credentials."""
    # Note that these are test private keys
    monkeypatch.setenv(
        "OPENGRADIENT_PRIVATE_KEY",
        "5ad9b639f7172b5b99936b1dfa1f95c04c4a9d2f4ea26019b2f3a65c4ecb3e59",
    )
