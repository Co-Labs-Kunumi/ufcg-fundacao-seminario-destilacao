class CostCalculator:

    # Mude os custos nesse método
    def __init__(self, price_per_million_input=1, price_per_million_output=3):
        # Custo de entrada
        self.input_price = price_per_million_input / 1_000_000
        # Custo de saída
        self.output_price = price_per_million_output / 1_000_000
        self.total_input = 0
        self.total_output = 0

    def add_tokens(self, input_tokens, output_tokens):
        self.total_input += input_tokens
        self.total_output += output_tokens

    def calculate(self):
        input_cost = self.total_input * self.input_price
        output_cost = self.total_output * self.output_price
        total_cost = input_cost + output_cost
        return {
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    def print_summary(self, total_time):
        result = self.calculate()
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        print(f"Total input tokens: {result['input_tokens']}")
        print(f"Total output tokens: {result['output_tokens']}")
        print(f"Input cost: ${result['input_cost']:.6f}")
        print(f"Output cost: ${result['output_cost']:.6f}")
        print(f"Estimated total cost: ${result['total_cost']:.6f}")
