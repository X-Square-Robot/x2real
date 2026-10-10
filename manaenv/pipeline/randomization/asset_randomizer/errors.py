from __future__ import annotations


class AssetRandomizerError(RuntimeError):
    """Base error for asset randomizer failures."""


class ConfigError(AssetRandomizerError):
    """Raised when authoring config cannot be parsed."""


class BackendError(AssetRandomizerError):
    """Raised when a backend cannot inspect or edit a scene."""


class BackendUnavailableError(BackendError):
    """Raised when optional backend dependencies are unavailable."""


class UnsupportedPolicyError(ConfigError):
    """Raised when a policy type is not registered."""
