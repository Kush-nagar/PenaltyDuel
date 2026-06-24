"""
Feature provider interface for tabular features now and video features later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class FeatureProvider(ABC):
    """Provider interface consumed by feature assembly code."""

    @abstractmethod
    def build(self, penalty_ids: Sequence[str]) -> pd.DataFrame:
        """Return features keyed by `penalty_id` in the requested order."""


class TabularFeatureProvider(FeatureProvider):
    """Serve precomputed tabular features keyed by penalty ID."""

    def __init__(self, features: pd.DataFrame) -> None:
        if "penalty_id" not in features.columns:
            raise ValueError("features must include a penalty_id column")
        self._features = features.copy()

    def build(self, penalty_ids: Sequence[str]) -> pd.DataFrame:
        requested = pd.DataFrame({"penalty_id": list(penalty_ids)})
        return requested.merge(self._features, on="penalty_id", how="left")


class VideoFeatureProvider(FeatureProvider):
    """
    Placeholder provider for the future video upgrade.

    Step 13 will replace this with real pose/ball/keeper-timing features.
    """

    def build(self, penalty_ids: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "penalty_id": list(penalty_ids),
                "video_is_missing": True,
            }
        )
