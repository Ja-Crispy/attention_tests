# my_attention_research_project/models/attention/__init__.py

# Only import modules that are fully implemented and ready for use.
from .vanilla_mha import MultiHeadAttention, ScaledDotProductAttention
from .ahc_attention import AHCAttention
# We will add MOAEAttention and PASAttention here once they are implemented.

# This line ensures that `from my_attention_research_project.models.attention import *`
# would import the currently available attention mechanisms.
__all__ = [
    "MultiHeadAttention", "ScaledDotProductAttention",
    "AHCAttention"
]