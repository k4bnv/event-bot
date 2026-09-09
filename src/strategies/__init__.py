from .absorption_reversal import AbsorptionReversalStrategy
from .adaptive_timing import AdaptiveTimingStrategy
from .ai_prompt import AIPromptStrategy
from .base import BaseStrategy, Signal, StrategyContext
from .breakout_retest import BreakoutRetestStrategy
from .fair_value_edge import FairValueEdgeStrategy
from .favorite_bias import FavoriteBiasStrategy
from .funding_skew import FundingSkewStrategy
from .mean_reversion import MeanReversionStrategy
from .orderbook_momentum import OrderbookMomentumStrategy
from .prior_window_momentum import PriorWindowMomentumStrategy
from .volatility_breakout import VolatilityBreakoutStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "breakout_retest": BreakoutRetestStrategy,
    "orderbook_momentum": OrderbookMomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "fair_value_edge": FairValueEdgeStrategy,
    "funding_skew": FundingSkewStrategy,
    "ai_prompt": AIPromptStrategy,
    "favorite_bias": FavoriteBiasStrategy,
    "prior_window_momentum": PriorWindowMomentumStrategy,
    "adaptive_timing": AdaptiveTimingStrategy,
    "absorption_reversal": AbsorptionReversalStrategy,
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
    "FavoriteBiasStrategy",
    "PriorWindowMomentumStrategy",
    "AdaptiveTimingStrategy",
    "AbsorptionReversalStrategy",
    "STRATEGY_REGISTRY",
]
