"""PolicyRemoteBase: transport-only component for ManaPolicy.

Optional component of :class:`~manaenv.policy.mana_policy.ManaPolicy`.
Pure transport + serialisation: take a request dict from
:meth:`PolicyImplementationBase.encode_request`, ship it to a remote
process, return the response dict. Does **not** know wire-field semantics.

Interface (policy-schema-contract.md):

  * ``get_actions(request: dict) -> response: dict``  abstractmethod
    Same name and same type as :meth:`PolicyImplementationBase.get_actions`
    so that ``ManaPolicy.load`` can bind either as the inference function
    transparently. The naming is multipolymorphic on purpose: at the impl
    layer the signature is ``(req) -> raw``; at the ManaPolicy layer it
    becomes ``(env, obs) -> Tensor``. The shared verb ``get_actions``
    keeps the call surface uniform across local / remote / chunked /
    non-chunked deployments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class PolicyRemoteBase(ABC):
    """Runtime base for PolicyRemote subclasses.

    Subclasses are resolved through policy remote discovery and implement
    :meth:`get_actions`. ``ManaPolicy.load`` resolves the
    inference function by selecting ``remote.get_actions`` when a remote
    is attached, else ``implementation.get_actions``.
    """

    def __init__(self) -> None:
        self.cfg: PolicyRemoteCFG | None = None
        self._is_loaded: bool = False

    def load(self, cfg: PolicyRemoteCFG) -> None:
        """Default: store *cfg* and mark loaded. Override to open transport
        resources (sockets, gRPC channels, etc.)."""
        self.cfg = cfg
        self._is_loaded = True

    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def get_actions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send *request* over the wire, return the response dict.

        Same signature as :meth:`PolicyImplementationBase.get_actions`.
        ``ManaPolicy`` binds one of the two as its inference function; the
        wire-codec impl (``encode_request`` + ``decode_response``) does
        not know which is in use.
        """

    def reset(self, env_ids: Any = None) -> None:
        """Optional per-env reset hook (no-op by default)."""

    def cleanup(self) -> None:
        """Optional resource release hook — close sockets/channels here."""


class PolicyRemoteBaseCFG:
    """Base marker for PolicyRemote CFG classes.

    Subclass CFGs **must** set ``class_type`` to their concrete Remote class.
    """


@dataclass
class PolicyRemoteCFG:
    """Default CFG for plain PolicyRemote subclasses.

    Subclasses extend this dataclass to add transport-specific
    parameters (e.g. ``server_address``, ``server_port``, ``retry_attempts``).
    The translator constructs a concrete CFG instance from a reflected remote
    section in the policy yaml.

    ``class_type`` defaults to ``None`` (this dataclass is the abstract base).
    Concrete subclass CFGs must set it to their concrete Remote class.
    """

    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type | None = None
