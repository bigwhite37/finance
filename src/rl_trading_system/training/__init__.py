"""训练系统模块"""

from .data_split_strategy import (
    DataSplitStrategy,
    TimeSeriesSplitStrategy,
    RollingWindowSplitStrategy,
    FixedSplitStrategy,
    SplitConfig,
    SplitResult,
    create_split_strategy,
    validate_split_quality
)

from .trainer import (
    RLTrainer,
    TrainingConfig,
    EarlyStopping,
    DrawdownEarlyStopping,
    TrainingMetrics
)

__all__ = [
    "DataSplitStrategy",
    "TimeSeriesSplitStrategy",
    "RollingWindowSplitStrategy", 
    "FixedSplitStrategy",
    "SplitConfig",
    "SplitResult",
    "create_split_strategy",
    "validate_split_quality",
    "RLTrainer",
    "TrainingConfig",
    "EarlyStopping",
    "DrawdownEarlyStopping",
    "TrainingMetrics"
]