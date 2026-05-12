"""
LARS: Learning-based Adaptive Residual Safety Adapter.

A lightweight safety module that wraps around NavDP/LoGoPlanner.
Freezes the original policy weights and adds a learned risk-assessment
layer with hard-rule safety guarantees.

This is NOT part of the original NavDP or LoGoPlanner method.
"""

from .depth_to_scan import DepthToScan
from .safety_shield import SafetyShield
from .action_adapter import ActionAdapter
from .logger import LARSLogger
from .metrics import LARSMetrics

# Torch-dependent modules are lazily imported to avoid requiring torch
# when only the rule-based components are needed.


def __getattr__(name):
    _lazy_imports = {
        "RiskResidualNet": ".risk_model",
        "ResidualAdapter": ".residual_adapter",
        "LARSRuntime": ".lars_runtime",
    }
    if name in _lazy_imports:
        import importlib
        mod = importlib.import_module(_lazy_imports[name], __package__)
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
