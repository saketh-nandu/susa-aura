"""
Unit tests for SUSA-1B CodeMind DataCleaner and Secret Scrubbing.
"""

from __future__ import annotations

import pytest
from susacodemind.data.cleaner import DataCleaner


@pytest.fixture
def cleaner() -> DataCleaner:
    return DataCleaner()


def test_secret_scrubbing_github_token(cleaner):
    """Verify GitHub personal access tokens are detected and redacted."""
    sample = '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ddeeff'
    cleaned, count = cleaner.scrub_secrets(sample)
    assert count == 1
    assert "[REDACTED_GITHUB_PAT]" in cleaned
    assert "11AABB" not in cleaned


def test_secret_scrubbing_aws_key(cleaner):
    """Verify AWS access keys are redacted."""
    sample = 'aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"'
    cleaned, count = cleaner.scrub_secrets(sample)
    assert count >= 1
    assert "[REDACTED_AWS_KEY]" in cleaned
    assert "AKIAIOSFODNN7EXAMPLE" not in cleaned


def test_secret_scrubbing_private_key(cleaner):
    """Verify RSA/OpenSSH private keys are detected and redacted."""
    sample = """
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Y1234567890abcdefABCDEF1234567890abcdefABCDEF
-----END RSA PRIVATE KEY-----
"""
    cleaned, count = cleaner.scrub_secrets(sample)
    assert count == 1
    assert "[REDACTED_PRIVATE_KEY]" in cleaned
    assert "MIIEow" not in cleaned


def test_secret_scrubbing_password_and_api_key(cleaner):
    """Verify assignment-based secrets and passwords are redacted."""
    sample = 'api_key = "abcdef1234567890_very_long_secret_token"\npassword = "SuperSecretPassword123!"'
    cleaned, count = cleaner.scrub_secrets(sample)
    assert count >= 1
    assert "[REDACTED_SECRET]" in cleaned or "[REDACTED_PASSWORD]" in cleaned


def test_normalization_and_line_breaks(cleaner):
    """Verify CRLF line breaks and null bytes are removed cleanly."""
    sample = "line 1\r\nline 2\r\x00line 3"
    norm = cleaner.normalize(sample)
    assert "\r" not in norm
    assert "\x00" not in norm
    assert norm == "line 1\nline 2\nline 3"


def test_minified_code_filtering(cleaner):
    """Verify minified single-line code is rejected."""
    minified = "var a=1;" * 500  # Very long single line (> 1000 chars)
    res = cleaner.clean_document(minified)
    assert res.is_valid is False
    assert "exceeds minification threshold" in res.rejection_reason


def test_valid_code_accepted(cleaner):
    """Verify standard clean Python code is approved."""
    code = """def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    res = cleaner.clean_document(code)
    assert res.is_valid is True
    assert res.secrets_redacted_count == 0
