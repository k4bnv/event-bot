from .ai_prompt import AIPromptStrategy
from .base import BaseStrategy, Signal, StrategyContext
from .breakout_retest import BreakoutRetestStrategy
from .fair_value_edge import FairValueEdgeStrategy
from .funding_skew import FundingSkewStrategy
from .mean_reversion import MeanReversionStrategy
from .orderbook_momentum import OrderbookMomentumStrategy
from .volatility_breakout import VolatilityBreakoutStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "breakout_retest": BreakoutRetestStrategy,
    "orderbook_momentum": OrderbookMomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "fair_value_edge": FairValueEdgeStrategy,
    "funding_skew": FundingSkewStrategy,
    "ai_prompt": AIPromptStrategy,
}

__all__ = [
    "BaseStrategy",
    "Signal",
    "StrategyContext",
    "BreakoutRetestStrategy",
    "OrderbookMomentumStrategy",
    "MeanReversionStrategy",
    "VolatilityBreakoutStrategy",
    "FairValueEdgeStrategy",
    "FundingSkewStrategy",
    "AIPromptStrategy",
    "STRATEGY_REGISTRY",
]
