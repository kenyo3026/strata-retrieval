"""Provider registry.

A frozen-dataclass registry (not an Enum) mapping provider name -> analyzer class.
`get_analyzer` resolves a name to its DocAnalyzer subclass; Main instantiates it.
Unknown names raise rather than falling back, since there is no default provider.
"""

from dataclasses import dataclass
from typing import Type

from .base import DocAnalyzer
from .mineru.analyzer import MinerUAnalyzer


@dataclass(frozen=True)
class ProviderType:
    MINERU: str = "MINERU"


@dataclass(frozen=True)
class ProviderRegistry:
    MINERU: Type[DocAnalyzer] = MinerUAnalyzer


def get_analyzer(name: str = ProviderType.MINERU) -> Type[DocAnalyzer]:
    analyzer = getattr(ProviderRegistry, name.upper(), None)
    if analyzer is None:
        raise ValueError(f"Unknown provider '{name}'")
    return analyzer
