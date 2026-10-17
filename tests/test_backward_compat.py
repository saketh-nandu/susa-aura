"""
Unit tests for SUSA AURA — Backward Compatibility Shims for CodeMind.
"""

from __future__ import annotations

import warnings


def test_susacodemind_import_and_deprecation_warning():
    """Verify importing susacodemind raises a helpful DeprecationWarning."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        import susacodemind
        assert hasattr(susacodemind, "__version__")
        assert any("deprecated" in str(w.message).lower() for w in recorded)


def test_model_shim_imports():
    """Verify legacy model imports resolve to AuraModel classes."""
    from susacodemind.model import (
        GroupedQueryAttention,
        KVCache,
        ModelConfig,
        RMSNorm,
        RotaryEmbedding,
        SusaCodeMind,
        SwiGLU,
    )
    from susacodemind.model.transformer import SusaCodeMind as SCMDirect
    from aura.model import AuraModel

    assert SusaCodeMind is AuraModel
    assert SCMDirect is AuraModel
    assert issubclass(SusaCodeMind, object)


def test_tokenizer_shim_imports():
    """Verify legacy tokenizer imports resolve to AuraTokenizer."""
    from susacodemind.tokenizer import SPECIAL_TOKENS, SusaTokenizer
    from susacodemind.tokenizer.bpe_tokenizer import SusaTokenizer as TokDirect
    from aura.tokenizer import AuraTokenizer

    assert SusaTokenizer is AuraTokenizer
    assert TokDirect is AuraTokenizer


def test_trainer_shim_imports():
    """Verify legacy trainer imports resolve to AuraTrainer."""
    from susacodemind.training import Trainer, TrainingConfig
    from susacodemind.training.trainer import Trainer as TrainerDirect
    from aura.training import AuraTrainer

    assert Trainer is AuraTrainer
    assert TrainerDirect is AuraTrainer


def test_inference_shim_imports():
    """Verify legacy generator imports resolve to AuraGenerator."""
    from susacodemind.inference import CodeGenerator
    from susacodemind.inference.generator import CodeGenerator as GenDirect
    from aura.inference import AuraGenerator

    assert CodeGenerator is AuraGenerator
    assert GenDirect is AuraGenerator


def test_agent_shim_imports():
    """Verify legacy agent imports resolve to AuraCore."""
    from susacodemind.agent import CodeMindAgent, WorkspaceInspector
    from susacodemind.agent.runtime import CodeMindAgent as AgentDirect
    from aura.core import AuraCore

    assert CodeMindAgent is AuraCore
    assert AgentDirect is AuraCore


def test_utils_shim_imports():
    """Verify legacy utility imports resolve correctly."""
    from susacodemind.utils import ResourceEstimate, SystemInfo, estimate_resources
    from susacodemind.utils.system_checker import SystemInfo as SIDirect
    from aura.utils import SystemInfo as AuraSI

    assert SystemInfo is AuraSI
    assert SIDirect is AuraSI


def test_evaluation_shim_imports():
    """Verify legacy evaluation imports resolve correctly."""
    from susacodemind.evaluation import CodeEvaluator, IsolatedCodeRunner
    from aura.evaluation import CodeEvaluator as AuraCE

    assert CodeEvaluator is AuraCE


def test_data_shim_imports():
    """Verify legacy data imports resolve correctly."""
    from susacodemind.data import DataCleaner, Deduplicator, MemoryMappedPretrainingDataset
    from aura.data import DataCleaner as AuraDC

    assert DataCleaner is AuraDC
