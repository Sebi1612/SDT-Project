"""Model-specific extraction adapters used by the multilingual analyses.

CodeBERT is deliberately not exposed here.  Its verified extraction path
continues to live in ``save_graph_info.py`` and ``save_word_embedding.py``.
"""

from .base import AdapterOutput, ModelAdapter, ModelSpec, TokenAlignmentError
from .registry import (
    MODEL_SPECS,
    create_model_adapter,
    get_model_spec,
    supported_model_names,
)

__all__ = [
    "AdapterOutput",
    "MODEL_SPECS",
    "ModelAdapter",
    "ModelSpec",
    "TokenAlignmentError",
    "create_model_adapter",
    "get_model_spec",
    "supported_model_names",
]
