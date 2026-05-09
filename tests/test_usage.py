"""
Tests for the usage module.
"""

import pytest

from cyberai import Usage


class TestUsage:
    """Tests for the Usage dataclass."""

    def test_default_values(self):
        """Usage should have all values defaulting to 0."""
        usage = Usage()
        assert usage.requests == 0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        """Usage can be created with custom values."""
        usage = Usage(requests=5, input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.requests == 5
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_add_combines_values(self):
        """Usage.add() should add another Usage's values."""
        usage1 = Usage(requests=2, input_tokens=100, output_tokens=50, total_tokens=150)
        usage2 = Usage(requests=3, input_tokens=200, output_tokens=100, total_tokens=300)

        usage1.add(usage2)

        assert usage1.requests == 5
        assert usage1.input_tokens == 300
        assert usage1.output_tokens == 150
        assert usage1.total_tokens == 450

    def test_add_to_empty_usage(self):
        """Adding to an empty Usage should work correctly."""
        usage1 = Usage()
        usage2 = Usage(requests=1, input_tokens=50, output_tokens=25, total_tokens=75)

        usage1.add(usage2)

        assert usage1.requests == 1
        assert usage1.input_tokens == 50
        assert usage1.output_tokens == 25
        assert usage1.total_tokens == 75

    def test_add_empty_usage(self):
        """Adding an empty Usage should not change values."""
        usage1 = Usage(requests=1, input_tokens=50, output_tokens=25, total_tokens=75)
        usage2 = Usage()

        usage1.add(usage2)

        assert usage1.requests == 1
        assert usage1.input_tokens == 50
        assert usage1.output_tokens == 25
        assert usage1.total_tokens == 75

    def test_add_multiple_times(self):
        """Adding multiple times should accumulate correctly."""
        usage = Usage()

        for i in range(3):
            other = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
            usage.add(other)

        assert usage.requests == 3
        assert usage.input_tokens == 30
        assert usage.output_tokens == 15
        assert usage.total_tokens == 45

    def test_partial_values(self):
        """Adding with some zero values should work correctly."""
        usage1 = Usage(requests=1, input_tokens=100, output_tokens=0, total_tokens=100)
        usage2 = Usage(requests=0, input_tokens=0, output_tokens=50, total_tokens=50)

        usage1.add(usage2)

        assert usage1.requests == 1
        assert usage1.input_tokens == 100
        assert usage1.output_tokens == 50
        assert usage1.total_tokens == 150
