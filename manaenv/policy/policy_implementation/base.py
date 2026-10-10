"""PolicyImplementationBase: protocol-specific single-frame wire codec.

Required component of :class:`~manaenv.policy.mana_policy.ManaPolicy`. Owns
the wire-codec for one protocol family (wallx / openpi / gr00t /
dp_image / replay / ...): encode a single-frame observation into a
request dict, decode a single response dict into an action chunk
tensor, and (for local-inference impls) optionally produce that
response dict from the request.

Interface — three methods, no chunk buffer, no remote handle, no
per-env state (policy-schema-contract.md):

  * ``encode_request(observation, env_idx=None) -> request: dict``  abstract
  * ``decode_response(raw, env=None, env_idx=None) -> chunk``       abstract
  * ``get_actions(request) -> raw: dict``            default raises;
    local-inference impls override (gr00t / dp_image / replay).

ManaPolicy owns the chunk pipeline (refill / wire hook / postproc /
ChunkBuffer) and routes inference through either ``remote.get_actions``
(when remote ``serving`` is set) or ``implementation.get_actions`` (local).
There is **no** local-vs-remote signal at the impl level — both modes
go through the same ``ManaPolicy.get_actions`` code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG


class PolicyImplementationBase(ABC):
    """Runtime base for policy implementation subclasses.

    Subclasses are resolved through policy implementation discovery and
    implement :meth:`encode_request` + :meth:`decode_response`.
    Local-inference impls additionally override :meth:`get_actions`;
    remote-only impls leave the default implementation in place (which
    raises) — ``ManaPolicy.load`` validates the absence of remote serving
    against this signal at load time, not at first inference.
    """

    def __init__(self) -> None:
        self.cfg: PolicyImplementationCFG | None = None
        self._embodiment: PolicyEmbodimentCFG | None = None
        self._is_loaded: bool = False

    def load(self, cfg: PolicyImplementationCFG) -> None:
        """Default: store *cfg* and mark loaded. Override to consume
        implementation-specific parameters (e.g. open a model checkpoint)."""
        self.cfg = cfg
        self._is_loaded = True

    def is_loaded(self) -> bool:
        return self._is_loaded

    def bind(self, embodiment: PolicyEmbodimentCFG) -> None:
        """Receive the PolicyEmbodimentCFG. Default just stores it; subclasses
        typically override to build their codec / transform tables."""
        self._embodiment = embodiment

    @abstractmethod
    def encode_request(self, observation: dict[str, Any], env_idx: int | None = None) -> dict[str, Any]:
        """Encode a single-frame simulator observation into a wire dict.

        The returned dict is fed verbatim to ``inference_fn(req)`` (either
        ``remote.get_actions`` or this impl's own :meth:`get_actions` for
        local inference). It is also the input to all
        ``client.wire_interceptors`` ``on_request`` hooks; interceptors
        may rewrite it before transport.
        """

    @abstractmethod
    def decode_response(
        self,
        raw: dict[str, Any],
        env: Any = None,
        env_idx: int | None = None,
    ) -> torch.Tensor | None:
        """Decode a single response dict into an action-chunk tensor.

        Returns ``(T, action_dim)`` for per-env refills (wallx / openpi /
        dp_image) or ``(num_envs, T, action_dim)`` for batched refills
        (gr00t). Returning ``None`` skips this refill cycle (e.g. the
        server returned no usable chunk).

        ``env`` is an optional sim-side handle (the IsaacLab environment
        instance) provided by ManaPolicy. wallx ``ik_eef_delta`` /
        ``ee_abs_ik`` modes need it to read the current EE pose for the
        delta computation; protocols whose decode is purely on the wire
        (openpi / dp_image / gr00t / replay) ignore it. Default ``None``
        keeps unit tests free from a real env handle.
        """

    def get_actions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inner-forward inference (default raises).

        Local-inference impls (gr00t / dp_image / replay) override this
        to call their inner model directly. Remote-only impls (wallx /
        openpi) leave the default in place; ``ManaPolicy.load`` then
        binds ``remote.get_actions`` as the inference function and
        validates that a remote serving config was provided. The signature here is
        kept identical to :meth:`PolicyRemoteBase.get_actions` so the
        two are interchangeable from ``ManaPolicy``'s point of view.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_actions(request) is not implemented; "
            "this implementation requires a remote CFG section in the policy yaml."
        )

    def observe(self, observation: dict[str, Any], env_idx: int | None = None) -> None:
        """Per-frame observation hook for temporal policies.

        Called by ManaPolicy on every control frame, including frames where
        no refill occurs. Stateless implementations (wallx/openpi/gr00t)
        leave this as a no-op. Temporal implementations that maintain a
        multi-frame history window (e.g. PolicySpaceImpl proxying DreamZero)
        override this to buffer frames between refill cycles.
        """

    def reset(self, env_ids: Any = None) -> None:
        """Optional per-env reset hook (no-op by default)."""

    def initialize_episode(self, env_ids: Any = None) -> None:
        """Prepare implementation state for a new evaluation episode.

        Generic implementations keep their existing reset hook. Protocols with
        an explicit episode lifecycle, such as Policy Space, override this
        method and use their native lifecycle operation.
        """
        self.reset(env_ids)

    def finalize_episode(
        self,
        env_ids: Any = None,
        outcomes: dict[int, bool] | None = None,
    ) -> None:
        """Release state associated with the episode that just ended."""

    def requires_serial_envs(self) -> bool:
        """Whether this implementation requires one rollout environment."""
        return False

    def cleanup(self) -> None:
        """Optional resource release hook (no-op by default)."""

    def resolve_instruction(self, env_idx: int | None = None) -> str:
        """Resolve the instruction text for one env.

        ``self._instruction`` is normally a single broadcast string but the
        benchmark/datagen runtime may push a **per-env list** (the active
        instruction store, one wording per env). Pick this env's entry when a
        list is in force; otherwise return the single string. Graceful
        degradation: an empty / out-of-range list entry falls back to the first
        non-empty entry, then to ``""``.
        """
        instruction = getattr(self, "_instruction", "")
        if isinstance(instruction, (list, tuple)):
            if not instruction:
                return ""
            if env_idx is not None and 0 <= env_idx < len(instruction) and instruction[env_idx]:
                return str(instruction[env_idx])
            for item in instruction:
                if item:
                    return str(item)
            return ""
        return str(instruction or "")

    def resolve_instruction_list(self, num_envs: int) -> list[str]:
        """Resolve a length-``num_envs`` per-env instruction list.

        Used by batched encoders (e.g. ``dp_image``) that build one entry per
        env. A per-env list in force is padded/truncated to ``num_envs``; a
        single string is broadcast; an empty instruction yields ``[]`` (caller
        omits the field, preserving today's behaviour)."""
        instruction = getattr(self, "_instruction", "")
        if isinstance(instruction, (list, tuple)):
            items = [str(item or "") for item in instruction]
            if not items:
                return []
            if len(items) >= num_envs:
                return items[:num_envs]
            return items + [items[-1]] * (num_envs - len(items))
        if instruction:
            return [str(instruction)] * num_envs
        return []


class PolicyImplementationBaseCFG:
    """Base marker for policy implementation CFG classes.

    Subclass CFGs **must** set ``class_type`` to their concrete
    implementation class.

    Each concrete CFG subclass MUST also declare ``YAML_DISPATCH_KEY``
    (``ClassVar[str]``). It is not a YAML dispatch key; it identifies the
    embodiment directory ``embodiments/<YAML_DISPATCH_KEY>/`` used when
    injecting the runtime ``PolicyEmbodimentCFG``.
    """

    YAML_DISPATCH_KEY: ClassVar[str] = ""


@dataclass
class PolicyImplementationCFG:
    """Default CFG for plain PolicyImplementation subclasses.

    Subclasses extend this dataclass to add implementation-specific
    parameters (e.g. ``instruction``, ``image_encode``, ``model_path``).
    The translator constructs a concrete CFG instance from the policy
    yaml's top-level inference parameters via ``_build_dataclass_from_dict``.

    Chunk-pipeline parameters (``action_chunk_length``, ``action_horizon``,
    ``interpolation_multiplier``, ``ee_pose_normalize``, ...) live in
    the policy yaml's ``client:`` section and are consumed by ManaPolicy,
    **not** here.

    ``class_type`` defaults to ``None`` (this dataclass is the abstract base).
    Concrete subclass CFGs must set it to their concrete Implementation class.
    """

    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type | None = None
