"""
Backward compatibility layer for susacodemind.model.
"""

from aura.model import (
    AuraModel,
    AuraOutput,
    GroupedQueryAttention,
    KVCache,
    ModelConfig,
    ModelOutput,
    RMSNorm,
    RotaryEmbedding,
    SusaCodeMind,
    SwiGLU,
    TransformerBlock,
    apply_rotary_pos_emb,
)
from aura.model.attention import GroupedQueryAttention as _GQA
from aura.model.config import ModelConfig as _MC
from aura.model.kv_cache import KVCache as _KC
from aura.model.rmsnorm import RMSNorm as _RN
from aura.model.rope import RotaryEmbedding as _RE, apply_rotary_pos_emb as _ARPE
from aura.model.swiglu import SwiGLU as _SG
from aura.model.transformer import SusaCodeMind as _SCM

__all__ = [
    "AuraModel",
    "AuraOutput",
    "GroupedQueryAttention",
    "KVCache",
    "ModelConfig",
    "ModelOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "SusaCodeMind",
    "SwiGLU",
    "TransformerBlock",
    "apply_rotary_pos_emb",
]
