# tests/unit/test_token_counter.py

import pytest

from capstone_ai_ziv.core.token_counter import (
    INPUT_USD_PER_MTOK,
    OUTPUT_USD_PER_MTOK,
    TokenCounter,
)

class TestTokenCounter:
    def test_tracks_cumulative_tokens(self):
        counter = TokenCounter()
        counter.add(input_tokens=100, output_tokens=50)
        counter.add(input_tokens=200, output_tokens=100)
        assert counter.total_input == 300
        assert counter.total_output == 150
        assert counter.total == 450

    def test_estimates_cost_arithmetic(self):
        counter = TokenCounter()
        counter.add(input_tokens=1_000_000, output_tokens=100_000)
        expected = INPUT_USD_PER_MTOK + (OUTPUT_USD_PER_MTOK * 0.1)
        assert counter.estimate_cost() == pytest.approx(expected)

    def test_input_and_output_rates_are_not_interchangeable(self):
        in_heavy = TokenCounter()
        in_heavy.add(input_tokens=1_000_000, output_tokens=0)
        out_heavy = TokenCounter()
        out_heavy.add(input_tokens=0, output_tokens=1_000_000)
        assert out_heavy.estimate_cost() > in_heavy.estimate_cost()

    def test_resets(self):
        counter = TokenCounter()
        counter.add(input_tokens=100, output_tokens=50)
        counter.reset()
        assert counter.total == 0
        assert counter.estimate_cost() == 0.0