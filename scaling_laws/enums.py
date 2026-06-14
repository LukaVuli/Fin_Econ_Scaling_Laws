"""Enumerations and constants used throughout the scaling_laws package."""

from __future__ import annotations

from enum import Enum
from typing import Union


class NormalizationType(str, Enum):
    """Enumeration of supported normalization types."""
    LAYER = "layer"
    BATCH = "batch"
    NONE = "none"


class ArchitectureMode(str, Enum):
    """Enumeration of supported architecture modes."""
    TAPERED = "tapered"
    FIXED_DEPTH = "fixed_depth"


class InitializerType(str, Enum):
    """Enumeration of supported weight initializers."""
    HE_NORMAL = "he_normal"
    GLOROT_UNIFORM = "glorot_uniform"


class PortfolioMode(str, Enum):
    """Enumeration of supported portfolio analysis modes."""
    PANEL = "panel"
    TS = "ts"


class ResumeMode(str, Enum):
    """Enumeration of supported result-file resume behaviors."""
    UPDATE_EXISTING = "update_existing"
    OVERWRITE = "overwrite"
    SKIP_EXISTING = "skip_existing"
    FAIL_IF_EXISTS = "fail_if_exists"

    @classmethod
    def coerce(cls, value: Union["ResumeMode", str]) -> "ResumeMode":
        """Coerce a ``ResumeMode`` enum value or its string ``.value`` into a ``ResumeMode``.

        Accepts only a ``ResumeMode`` instance or one of the four enum string
        values (``"update_existing"``, ``"overwrite"``, ``"skip_existing"``,
        ``"fail_if_exists"``). Anything else raises ``ValueError``.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for mode in cls:
                if value == mode.value:
                    return mode
        raise ValueError(
            f"Unsupported resume mode: {value!r}. Expected a ResumeMode or one of "
            f"{[mode.value for mode in cls]!r}."
        )

    def __str__(self) -> str:
        return self.value


class SplitMode(str, Enum):
    """Enumeration of supported DataFrame split strategies."""
    AUTO = "auto"
    DATE_CUTOFFS = "date_cutoffs"
    DATE_PROPORTIONS = "date_proportions"
    MASKS = "masks"
    PRE_SPLIT = "pre_split"


class MissingDataPolicy(str, Enum):
    """Enumeration of supported missing-data policies."""
    DROP_ANY = "drop_any"
    DROP_TARGET_ONLY = "drop_target_only"
    ERROR = "error"
    IMPUTE_MEAN = "impute_mean"


VALID_BENCHMARK_MODES = {
    "historical_mean",
    "historical_mean_updating",
    "ar1",
    "ar1_updating",
}
