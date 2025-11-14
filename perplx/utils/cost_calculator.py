"""
Cost Calculator - Utilities for calculating AWS Bedrock API costs based on token usage

Supports Claude 3.5 Haiku pricing:
- Regular pricing: $0.00025 per 1K input tokens, $0.00125 per 1K output tokens
- Batch pricing: $0.000125 per 1K input tokens, $0.000625 per 1K output tokens
"""

from typing import Dict, Optional
from decimal import Decimal, ROUND_HALF_UP


class BedrockCostCalculator:
    """
    Calculate AWS Bedrock API costs based on token usage
    
    Pricing for Claude 3.5 Haiku:
    - Regular: $0.00025/1K input, $0.00125/1K output
    - Batch: $0.000125/1K input, $0.000625/1K output
    """
    
    # Claude 3.5 Haiku Pricing (per 1,000 tokens)
    PRICING_REGULAR = {
        "input": Decimal("0.00025"),   # $0.00025 per 1,000 input tokens
        "output": Decimal("0.00125")   # $0.00125 per 1,000 output tokens
    }
    
    PRICING_BATCH = {
        "input": Decimal("0.000125"),   # $0.000125 per 1,000 input tokens
        "output": Decimal("0.000625")  # $0.000625 per 1,000 output tokens
    }
    
    def __init__(self, use_batch_pricing: bool = False):
        """
        Initialize cost calculator
        
        Args:
            use_batch_pricing: If True, use batch pricing (cheaper). Default: False (regular pricing)
        """
        self.use_batch_pricing = use_batch_pricing
        self.pricing = self.PRICING_BATCH if use_batch_pricing else self.PRICING_REGULAR
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        round_to: int = 6
    ) -> Dict[str, float]:
        """
        Calculate cost based on input and output token counts
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            round_to: Number of decimal places to round to (default: 6)
            
        Returns:
            Dictionary with cost breakdown:
            {
                "input_cost": float,
                "output_cost": float,
                "total_cost": float,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "pricing_type": str
            }
        """
        # Convert to Decimal for precise calculations
        input_tokens_decimal = Decimal(str(input_tokens))
        output_tokens_decimal = Decimal(str(output_tokens))
        
        # Calculate costs (price is per 1,000 tokens)
        input_cost = (input_tokens_decimal / Decimal("1000")) * self.pricing["input"]
        output_cost = (output_tokens_decimal / Decimal("1000")) * self.pricing["output"]
        total_cost = input_cost + output_cost
        
        # Round to specified decimal places
        input_cost_rounded = float(input_cost.quantize(
            Decimal('0.1') ** round_to,
            rounding=ROUND_HALF_UP
        ))
        output_cost_rounded = float(output_cost.quantize(
            Decimal('0.1') ** round_to,
            rounding=ROUND_HALF_UP
        ))
        total_cost_rounded = float(total_cost.quantize(
            Decimal('0.1') ** round_to,
            rounding=ROUND_HALF_UP
        ))
        
        return {
            "input_cost": input_cost_rounded,
            "output_cost": output_cost_rounded,
            "total_cost": total_cost_rounded,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "pricing_type": "batch" if self.use_batch_pricing else "regular"
        }
    
    def calculate_cost_from_response(
        self,
        response_body: Dict,
        round_to: int = 6
    ) -> Dict[str, float]:
        """
        Calculate cost directly from Bedrock API response body
        
        Args:
            response_body: Parsed JSON response from Bedrock API
            round_to: Number of decimal places to round to (default: 6)
            
        Returns:
            Dictionary with cost breakdown (same format as calculate_cost)
        """
        try:
            usage = response_body.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            
            return self.calculate_cost(
                int(input_tokens),
                int(output_tokens),
                round_to=round_to
            )
        except (KeyError, ValueError, TypeError) as e:
            # Return zero cost if unable to extract tokens
            return {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "pricing_type": "batch" if self.use_batch_pricing else "regular",
                "error": str(e)
            }
    
    def format_cost_string(
        self,
        input_tokens: int,
        output_tokens: int,
        include_breakdown: bool = True
    ) -> str:
        """
        Format cost as a human-readable string
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            include_breakdown: If True, include input/output breakdown
            
        Returns:
            Formatted cost string
        """
        cost_data = self.calculate_cost(input_tokens, output_tokens)
        
        pricing_type_str = "batch" if self.use_batch_pricing else "regular"
        
        if include_breakdown:
            return (
                f"💰 Cost ({pricing_type_str} pricing): "
                f"Input: ${cost_data['input_cost']:.6f} ({input_tokens:,} tokens), "
                f"Output: ${cost_data['output_cost']:.6f} ({output_tokens:,} tokens), "
                f"Total: ${cost_data['total_cost']:.6f}"
            )
        else:
            return f"💰 Total Cost ({pricing_type_str} pricing): ${cost_data['total_cost']:.6f}"
    
    def get_pricing_info(self) -> Dict[str, str]:
        """
        Get current pricing information
        
        Returns:
            Dictionary with pricing details
        """
        pricing_type = "batch" if self.use_batch_pricing else "regular"
        pricing = self.pricing
        
        return {
            "pricing_type": pricing_type,
            "input_price_per_1k": f"${pricing['input']:.6f}",
            "output_price_per_1k": f"${pricing['output']:.6f}",
            "model": "Claude 3.5 Haiku"
        }


# Convenience functions for easy usage
def calculate_bedrock_cost(
    input_tokens: int,
    output_tokens: int,
    use_batch_pricing: bool = False,
    round_to: int = 6
) -> Dict[str, float]:
    """
    Convenience function to calculate Bedrock cost
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        use_batch_pricing: Use batch pricing if True
        round_to: Decimal places to round to
        
    Returns:
        Dictionary with cost breakdown
        
    Example:
        >>> cost = calculate_bedrock_cost(1000, 500)
        >>> print(f"Total cost: ${cost['total_cost']:.6f}")
    """
    calculator = BedrockCostCalculator(use_batch_pricing=use_batch_pricing)
    return calculator.calculate_cost(input_tokens, output_tokens, round_to=round_to)


def format_cost(
    input_tokens: int,
    output_tokens: int,
    use_batch_pricing: bool = False,
    include_breakdown: bool = True
) -> str:
    """
    Convenience function to format cost as string
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        use_batch_pricing: Use batch pricing if True
        include_breakdown: Include input/output breakdown
        
    Returns:
        Formatted cost string
        
    Example:
        >>> print(format_cost(1000, 500))
        💰 Cost (regular pricing): Input: $0.000250 (1,000 tokens), Output: $0.000625 (500 tokens), Total: $0.000875
    """
    calculator = BedrockCostCalculator(use_batch_pricing=use_batch_pricing)
    return calculator.format_cost_string(input_tokens, output_tokens, include_breakdown=include_breakdown)

