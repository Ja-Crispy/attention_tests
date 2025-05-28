import torch
from abc import ABC, abstractmethod

class BaseCombinationStrategy(ABC):
    @abstractmethod
    def apply(self, local_context: torch.Tensor, global_context_summaries: torch.Tensor, chunk_size: int) -> torch.Tensor:
        pass

class C1BroadcastAddCombinationStrategy(BaseCombinationStrategy):
    def apply(self, local_context: torch.Tensor, global_context_summaries: torch.Tensor, chunk_size: int) -> torch.Tensor:
        if global_context_summaries is None:
            raise ValueError("Global context summaries are None, cannot apply C1_broadcast_add combination.")
        if local_context is None:
            raise ValueError("Local context is None, cannot apply C1_broadcast_add combination.")
        
        # Ensure global_context_summaries is (B, N, D) before unsqueezing
        # local_context is (B, N, C, D)
        if global_context_summaries.ndim != 3:
            raise ValueError(f"Expected global_context_summaries to be 3D (B, N, D), got {global_context_summaries.ndim}D")

        # Expand global_context_summaries from (B, N, D) to (B, N, 1, D) and then to (B, N, C, D)
        expanded_global_context = global_context_summaries.unsqueeze(2).expand(-1, -1, chunk_size, -1)
        combined_context = local_context + expanded_global_context
        return combined_context

def get_combination_strategy(config: dict) -> BaseCombinationStrategy:
    strategy_type = config.get('type')
    if strategy_type == "C1_broadcast_add":
        return C1BroadcastAddCombinationStrategy()
    elif strategy_type is None:
        # This case should ideally be handled by a high-level configuration check or
        # the AHCAttention class might decide not to apply any combination.
        # Raising an error here ensures the config explicitly states the intent.
        raise ValueError("Combination strategy type cannot be None in config. If no combination is intended, this implies a different model path or explicit 'None' strategy type.")
    else:
        raise ValueError(f"Unsupported combination strategy type: {strategy_type}")
