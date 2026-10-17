"""
SUSA AURA — Dataset Sanitization & Secret Scrubbing Subsystem.

Detects and redacts secrets (API keys, private keys, passwords) and filters
corrupt, minified, or malformed source code files to enforce dataset safety.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Regex patterns for detecting and scrubbing credentials and secrets
SECRET_PATTERNS = [
    # GitHub Personal Access Tokens and OAuth tokens
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,255}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "[REDACTED_GITHUB_PAT]"),
    # AWS Access Key ID
    (re.compile(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}"), "[REDACTED_AWS_KEY]"),
    # RSA / OpenSSH Private Keys
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    # Generic API Keys / Tokens in assignments (e.g., api_key = "...", token: "...")
    (re.compile(r"""(?i)(['"]?(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key)['"]?\s*[:=]\s*['"])[a-zA-Z0-9_\-\.]{20,}(['"])"""), r"\g<1>[REDACTED_SECRET]\g<2>"),
    # Generic Password assignments
    (re.compile(r"""(?i)(['"]?password['"]?\s*[:=]\s*['"])[^'"]{8,}(['"])"""), r"\g<1>[REDACTED_PASSWORD]\g<2>"),
    # Database connection strings with credentials (e.g. postgresql://user:pass@host:port/db)
    (re.compile(r"://[^:\s]+:[^@\s]+@[a-zA-Z0-9\.\-]+(?::\d+)?/[^\s]+"), "://[REDACTED_USER]:[REDACTED_PASS]@[HOST]/[DB]"),
]


@dataclass
class FilterResult:
    """Outcome of filtering and sanitizing a code document."""

    is_valid: bool
    text: str
    rejection_reason: Optional[str] = None
    secrets_redacted_count: int = 0


class DataCleaner:
    """Sanitizes text and code, scrubs secrets, and applies quality filters."""

    def __init__(
        self,
        max_file_size_bytes: int = 1_000_000,  # 1 MB max file size
        max_avg_line_length: int = 1_000,      # Filter minified files (JS/CSS bundles)
        min_char_count: int = 20,              # Filter empty / trivial files
    ) -> None:
        self.max_file_size_bytes = max_file_size_bytes
        self.max_avg_line_length = max_avg_line_length
        self.min_char_count = min_char_count

    def normalize(self, text: str) -> str:
        """Applies NFKC Unicode normalization and standardizes line endings."""
        # Standardize line breaks: \r\n -> \n, \r -> \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove null bytes
        text = text.replace("\x00", "")
        # Apply standard NFKC Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        return text

    def scrub_secrets(self, text: str) -> Tuple[str, int]:
        """Detects and replaces secrets with redact placeholders.

        Returns:
            Tuple of (sanitized_text, total_secrets_redacted)
        """
        redacted_count = 0
        cleaned_text = text

        for pattern, replacement in SECRET_PATTERNS:
            cleaned_text, matches = pattern.subn(replacement, cleaned_text)
            redacted_count += matches

        return cleaned_text, redacted_count

    def clean_document(self, text: str) -> FilterResult:
        """Runs the complete cleaning, filtering, and secret redaction pipeline."""
        if not text or len(text.strip()) == 0:
            return FilterResult(is_valid=False, text="", rejection_reason="Empty document")

        # Byte length check
        byte_len = len(text.encode("utf-8", errors="ignore"))
        if byte_len > self.max_file_size_bytes:
            return FilterResult(
                is_valid=False,
                text="",
                rejection_reason=f"File size ({byte_len} bytes) exceeds limit ({self.max_file_size_bytes} bytes)",
            )

        # Normalization
        norm_text = self.normalize(text)

        # Minimum content check
        if len(norm_text.strip()) < self.min_char_count:
            return FilterResult(
                is_valid=False,
                text="",
                rejection_reason=f"Character count ({len(norm_text.strip())}) below minimum ({self.min_char_count})",
            )

        # Minified code heuristic check (average line length)
        lines = norm_text.splitlines()
        if len(lines) > 0:
            avg_line_len = sum(len(line) for line in lines) / len(lines)
            if avg_line_len > self.max_avg_line_length:
                return FilterResult(
                    is_valid=False,
                    text="",
                    rejection_reason=f"Average line length ({avg_line_len:.1f}) exceeds minification threshold ({self.max_avg_line_length})",
                )

        # Secret scrubbing
        sanitized_text, secrets_count = self.scrub_secrets(norm_text)

        return FilterResult(
            is_valid=True,
            text=sanitized_text,
            secrets_redacted_count=secrets_count,
        )
