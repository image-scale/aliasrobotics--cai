"""
Usage tracking for LLM API calls.
"""

from dataclasses import dataclass


@dataclass
class Usage:
    """Tracks token usage and request counts for LLM API calls."""

    requests: int = 0
    """Total number of requests made to the LLM API."""

    input_tokens: int = 0
    """Total input tokens sent across all requests."""

    output_tokens: int = 0
    """Total output tokens received across all requests."""

    total_tokens: int = 0
    """Total tokens (input + output) across all requests."""

    def add(self, other: "Usage") -> None:
        """Add another Usage instance's values to this one.

        Args:
            other: The Usage instance to add.
        """
        self.requests += other.requests if other.requests else 0
        self.input_tokens += other.input_tokens if other.input_tokens else 0
        self.output_tokens += other.output_tokens if other.output_tokens else 0
        self.total_tokens += other.total_tokens if other.total_tokens else 0
