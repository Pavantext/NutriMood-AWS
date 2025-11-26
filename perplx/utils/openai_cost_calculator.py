"""
OpenAI Cost Calculator - Utilities for calculating OpenAI API costs based on token usage

Supports GPT-4o-mini pricing (per 1M tokens):
- Input: $0.15 per 1M tokens
- Cached Input: $0.075 per 1M tokens
- Output: $0.60 per 1M tokens
"""

from typing import Dict, Optional
from decimal import Decimal, ROUND_HALF_UP


class OpenAICostCalculator:
    """
    Calculate OpenAI API costs based on token usage
    
    Pricing for GPT-4o-mini (per 1M tokens):
    - Input: $0.15
    - Cached Input: $0.075
    - Output: $0.60
    """
    
    # GPT-4o-mini Pricing (per 1,000,000 tokens)
    PRICING_PER_1M = {
        "input": Decimal("0.15"),         # $0.15 per 1M input tokens
        "cached_input": Decimal("0.075"), # $0.075 per 1M cached input tokens
        "output": Decimal("0.60")         # $0.60 per 1M output tokens
    }
    
    # Convert to per-token pricing for easier calculations
    PRICING_PER_TOKEN = {
        "input": Decimal("0.15") / Decimal("1000000"),         # $0.00000015 per token
        "cached_input": Decimal("0.075") / Decimal("1000000"), # $0.000000075 per token
        "output": Decimal("0.60") / Decimal("1000000")         # $0.0000006 per token
    }
    
    def __init__(self):
        """Initialize OpenAI cost calculator for GPT-4o-mini"""
        self.model = "gpt-4o-mini"
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        round_to: int = 8
    ) -> Dict[str, float]:
        """
        Calculate cost based on input and output token counts
        
        Args:
            input_tokens: Number of input tokens used (non-cached)
            output_tokens: Number of output tokens used
            cached_input_tokens: Number of cached input tokens (optional)
            round_to: Number of decimal places to round to (default: 8)
            
        Returns:
            Dictionary with cost breakdown:
            {
                "input_cost": float,
                "cached_input_cost": float,
                "output_cost": float,
                "total_cost": float,
                "input_tokens": int,
                "cached_input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "model": str
            }
        """
        # Convert to Decimal for precise calculations
        input_tokens_decimal = Decimal(str(input_tokens))
        cached_input_tokens_decimal = Decimal(str(cached_input_tokens))
        output_tokens_decimal = Decimal(str(output_tokens))
        
        # Calculate costs
        input_cost = input_tokens_decimal * self.PRICING_PER_TOKEN["input"]
        cached_input_cost = cached_input_tokens_decimal * self.PRICING_PER_TOKEN["cached_input"]
        output_cost = output_tokens_decimal * self.PRICING_PER_TOKEN["output"]
        total_cost = input_cost + cached_input_cost + output_cost
        
        # Round to specified decimal places
        def round_decimal(value: Decimal) -> float:
            return float(value.quantize(
                Decimal('0.1') ** round_to,
                rounding=ROUND_HALF_UP
            ))
        
        return {
            "input_cost": round_decimal(input_cost),
            "cached_input_cost": round_decimal(cached_input_cost),
            "output_cost": round_decimal(output_cost),
            "total_cost": round_decimal(total_cost),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + cached_input_tokens + output_tokens,
            "model": self.model
        }
    
    def calculate_cost_simple(
        self,
        input_tokens: int,
        output_tokens: int,
        round_to: int = 8
    ) -> float:
        """
        Calculate total cost (simplified - returns just the total)
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            round_to: Number of decimal places to round to (default: 8)
            
        Returns:
            Total cost in USD as float
        """
        cost_data = self.calculate_cost(input_tokens, output_tokens, 0, round_to)
        return cost_data["total_cost"]
    
    def format_cost_string(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        include_breakdown: bool = True
    ) -> str:
        """
        Format cost as a human-readable string
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_input_tokens: Number of cached input tokens
            include_breakdown: If True, include input/output breakdown
            
        Returns:
            Formatted cost string
        """
        cost_data = self.calculate_cost(input_tokens, output_tokens, cached_input_tokens)
        
        if include_breakdown:
            result = f"💰 OpenAI Cost ({self.model}): "
            result += f"Input: ${cost_data['input_cost']:.8f} ({input_tokens:,} tokens), "
            if cached_input_tokens > 0:
                result += f"Cached: ${cost_data['cached_input_cost']:.8f} ({cached_input_tokens:,} tokens), "
            result += f"Output: ${cost_data['output_cost']:.8f} ({output_tokens:,} tokens), "
            result += f"Total: ${cost_data['total_cost']:.8f}"
            return result
        else:
            return f"💰 Total Cost ({self.model}): ${cost_data['total_cost']:.8f}"
    
    def get_pricing_info(self) -> Dict[str, str]:
        """
        Get current pricing information
        
        Returns:
            Dictionary with pricing details
        """
        return {
            "model": self.model,
            "input_price_per_1m": f"${self.PRICING_PER_1M['input']}",
            "cached_input_price_per_1m": f"${self.PRICING_PER_1M['cached_input']}",
            "output_price_per_1m": f"${self.PRICING_PER_1M['output']}"
        }


# Convenience functions for easy usage
def calculate_openai_cost(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    round_to: int = 8
) -> Dict[str, float]:
    """
    Convenience function to calculate OpenAI cost
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_input_tokens: Number of cached input tokens
        round_to: Decimal places to round to
        
    Returns:
        Dictionary with cost breakdown
        
    Example:
        >>> cost = calculate_openai_cost(1000, 500)
        >>> print(f"Total cost: ${cost['total_cost']:.8f}")
    """
    calculator = OpenAICostCalculator()
    return calculator.calculate_cost(input_tokens, output_tokens, cached_input_tokens, round_to)


def get_total_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Get just the total cost for given tokens
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
        
    Example:
        >>> cost = get_total_cost(1000, 500)
        >>> print(f"${cost:.8f}")
    """
    calculator = OpenAICostCalculator()
    return calculator.calculate_cost_simple(input_tokens, output_tokens)


def format_openai_cost(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    include_breakdown: bool = True
) -> str:
    """
    Convenience function to format cost as string
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_input_tokens: Number of cached input tokens
        include_breakdown: Include input/output breakdown
        
    Returns:
        Formatted cost string
        
    Example:
        >>> print(format_openai_cost(1000, 500))
        💰 OpenAI Cost (gpt-4o-mini): Input: $0.00015000 (1,000 tokens), Output: $0.00030000 (500 tokens), Total: $0.00045000
    """
    calculator = OpenAICostCalculator()
    return calculator.format_cost_string(input_tokens, output_tokens, cached_input_tokens, include_breakdown)

