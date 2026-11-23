# Pricing changes - check the pricing page before trusting any number this
# produces: https://platform.claude.com/docs/en/about-claude/pricing
INPUT_USD_PER_MTOK = 2.00
OUTPUT_USD_PER_MTOK = 10.00


class TokenCounter:
    COST_PER_INPUT_TOKEN = INPUT_USD_PER_MTOK / 1_000_000
    COST_PER_OUTPUT_TOKEN = OUTPUT_USD_PER_MTOK / 1_000_000

    def __init__(self):
        self.total_input = 0
        self.total_output = 0

    @property
    def total(self) -> int:
        return self.total_input + self.total_output

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input += input_tokens
        self.total_output += output_tokens

    def estimate_cost(self) -> float:
        return (
            self.total_input * self.COST_PER_INPUT_TOKEN
            + self.total_output * self.COST_PER_OUTPUT_TOKEN
        )

    def reset(self) -> None:
        self.total_input = 0
        self.total_output = 0