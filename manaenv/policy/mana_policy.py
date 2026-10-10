"""ManaPolicy: unified shell pattern policy class with embedded chunk pipeline.

Analogous to ``ManaTask``: a single concrete class composed at runtime from
three orthogonal components plus an optional client-side segment:

  * ``implementation`` (PolicyImplementation)     required — wire codec
  * ``remote``         (PolicyRemote | None)      optional — transport
  * ``embodiment``     (PolicyEmbodimentCFG)      required — model<->robot map
  * ``client``         (PolicyClientCFG | None)   optional — chunk pipeline

Interface contract (PLAN.md §10.4 + §10.5):

  * ``ManaPolicy.get_actions(env, observation) -> Tensor`` is the **only**
    external inference entry point. Chunk buffering, scheduler dispatch,
    wire-interceptor application and chunk postprocessing stay in the
    client-side pipeline — implementations stay single-frame and stateless.
  * ``PolicyImplementationBase`` exposes the wire codec (``encode_request``,
    ``decode_response``) and an inner ``get_actions(request) -> raw`` that
    only local-inference impls override (gr00t / dp_image / replay).
  * ``PolicyRemoteBase.get_actions(request) -> raw`` has the same signature
    as ``PolicyImplementationBase.get_actions``; ``ManaPolicy.load`` binds
    one of the two as the inference function. There is **no** local-vs-
    remote signal at the call surface — both go through the same code path.
  * ``PolicyEmbodimentCFG`` is a single dataclass (CFG = runtime, no
    Instance class) describing the model<->embodiment field map +
    ``action_mode`` + ``action_dim``. Looked up by DataCollectionFactory
    and consumed via :meth:`PolicyImplementationBase.bind`.
  * ``PolicyClientCFG`` carries the reflected client-side pipeline as
    plugin CFGs: a refill policy CFG (from
    ``manaenv.policy.refill_policies``), wire interceptor CFGs (from
    ``manaenv.policy.wire_interceptors``), chunk postprocessor CFGs
    (from ``manaenv.policy.chunk_postprocessors``), and a chunk scheduler
    CFG (from ``manaenv.policy.chunk_scheduler``). Each plugin is
    instantiated via ``cfg.class_type()`` + ``load(cfg)`` inside
    :meth:`ManaPolicy.load`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch

from manaenv.policy.chunk_connectors import connect_action_chunk
from manaenv.policy.chunk_postprocessors import ChunkPostprocessorBase, ChunkPostprocessorBaseCFG
from manaenv.policy.chunk_scheduler.base import REFILL_CONTEXT_KEY, ChunkSchedulerBase, ChunkSchedulerBaseCFG
from manaenv.policy.chunk_scheduler.sync import SyncChunkSchedulerCFG
from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG
from manaenv.policy.policy_implementation.base import PolicyImplementationBase, PolicyImplementationCFG
from manaenv.policy.policy_remote.base import PolicyRemoteCFG
from manaenv.policy.refill_policies import PerEnvIndependentCFG, RefillPolicyBase, RefillPolicyBaseCFG
from manaenv.policy.wire_interceptors import WireInterceptorBase, WireInterceptorBaseCFG
from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import normalize_env_ids

if TYPE_CHECKING:
    from manaenv.policy.policy_remote.base import PolicyRemoteBase


@dataclass
class ChunkBuffer:
    """Per-env action chunk buffer.

    ``frames[env_idx]`` holds the remaining ``(T, action_dim)`` tensor for
    that env. ``pop_step`` consumes the leading row per env and returns a
    stacked ``(num_envs, action_dim)`` tensor. An empty buffer holds the
    previous command instead of emitting zeros, so a late refill cannot
    yank a still-running env toward the default pose. ``fill`` replaces an
    env's buffer wholesale; ``needs_refill`` returns True when the env's
    buffer has fewer than ``threshold`` rows left (or no buffer at all).
    """

    frames: dict[int, torch.Tensor] = field(default_factory=dict)
    last_commands: dict[int, torch.Tensor] = field(default_factory=dict)

    def needs_refill(self, env_idx: int, threshold: int) -> bool:
        chunk = self.frames.get(env_idx)
        if chunk is None:
            return True
        return int(chunk.shape[0]) < max(1, int(threshold))

    def fill(self, env_idx: int, chunk: torch.Tensor) -> None:
        self.frames[int(env_idx)] = chunk

    def remaining_len(self, env_idx: int) -> int:
        chunk = self.frames.get(int(env_idx))
        if chunk is None:
            return 0
        return int(chunk.shape[0])

    def pop_step(self, num_envs: int, action_dim: int, device: Any) -> torch.Tensor:
        out = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=device)
        held_envs: list[int] = []
        empty_no_hold: list[int] = []
        for env_idx in range(num_envs):
            chunk = self.frames.get(env_idx)
            if chunk is not None and int(chunk.shape[0]) > 0:
                step_action = chunk[0].to(device=device, dtype=torch.float32)
                out[env_idx] = step_action
                self.last_commands[env_idx] = step_action.detach().clone()
                self.frames[env_idx] = chunk[1:]
                continue
            last = self.last_commands.get(env_idx)
            if last is None or int(last.numel()) != action_dim:
                empty_no_hold.append(env_idx)
                continue
            out[env_idx] = last.to(device=device, dtype=torch.float32)
            held_envs.append(env_idx)
        if held_envs or (empty_no_hold and self.last_commands):
            remain = {idx: self.remaining_len(idx) for idx in range(num_envs)}
            last_l2 = {
                idx: round(float(cmd.detach().float().norm().item()), 4) for idx, cmd in self.last_commands.items()
            }
            manaprint(
                "WARNING: [ChunkBuffer] pop_step empty-buffer fallback "
                f"held={held_envs} empty_no_hold={empty_no_hold} remain={remain} "
                f"last_l2={last_l2} wall={time.time():.3f}"
            )
        return out

    def get_last_command(self, env_idx: int) -> torch.Tensor | None:
        return self.last_commands.get(int(env_idx))

    def clear(self, env_ids: Any = None) -> None:
        ids = normalize_env_ids(env_ids)
        if ids is None:
            self.frames.clear()
            self.last_commands.clear()
            return
        for i in ids:
            env_i = int(i)
            self.frames.pop(env_i, None)
            self.last_commands.pop(env_i, None)


@dataclass
class PolicyClientCFG:
    """Reflected ``client:`` segment of a policy yaml (PLAN.md §10.5).

    Every sub-plugin is declared with its explicit CFG key in YAML and
    reflected through the global lazy registry, e.g.::

        client:
          chunk_executor:
            per_env_independent: {execute_horizon: 32, refill_threshold: 1}
          chunk_postprocessors:
          - temporal_interpolate: {multiplier: 1.5}
          chunk_scheduler:
            async_chunk_scheduler: {}
          connect_chunks: true

    Default: no executor CFG (-> per-env independent, horizon 1), no
    hooks / postprocs — equivalent to "every step is a single-frame
    inference".
    """

    chunk_executor: RefillPolicyBaseCFG | None = None
    wire_interceptors: list[WireInterceptorBaseCFG] = field(default_factory=list)
    chunk_postprocessors: list[ChunkPostprocessorBaseCFG] = field(default_factory=list)
    chunk_scheduler: ChunkSchedulerBaseCFG = field(default_factory=SyncChunkSchedulerCFG)
    connect_chunks: bool = False

    @property
    def execute_horizon(self) -> int:
        if self.chunk_executor is None:
            return 1
        return max(1, int(self.chunk_executor.execute_horizon))

    @property
    def refill_threshold(self) -> int:
        if self.chunk_executor is None:
            return 1
        return max(1, int(self.chunk_executor.refill_threshold))


class PolicyBase:
    """Legacy policy parent kept for compatibility with older tests.

    The single concrete subclass is :class:`ManaPolicy`. Do **not** add
    new policies by subclassing this — register a new
    ``PolicyImplementation`` (and optionally ``PolicyRemote``) instead.
    """


class PolicyBaseCFG:
    """Legacy policy CFG parent kept for compatibility with older tests."""


class ManaPolicy:
    """Unified concrete Policy class. See module docstring for the
    composition contract.

    ``ManaPolicy`` owns the chunk pipeline at runtime: it picks an
    inference function (remote.get_actions when remote ``serving`` is set,
    impl.get_actions otherwise), drives the refill function, applies
    wire interceptors and chunk postprocessors, and serves one row per
    env per step from a per-env :class:`ChunkBuffer`.
    """

    def __init__(self) -> None:
        self.cfg: ManaPolicyCFG | None = None
        self.implementation: PolicyImplementationBase | None = None
        self.remote: PolicyRemoteBase | None = None
        self.embodiment: PolicyEmbodimentCFG | None = None
        self._client: PolicyClientCFG = PolicyClientCFG()
        self._buffer: ChunkBuffer = ChunkBuffer()
        self._get_actions_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._is_loaded: bool = False
        self._chunk_scheduler: ChunkSchedulerBase | None = None
        self._refill_policy: RefillPolicyBase | None = None
        self._wire_hooks: list[WireInterceptorBase] = []
        self._postproc_fns: list[ChunkPostprocessorBase] = []
        self._inference_lock: Lock = Lock()

    def load(self, cfg: ManaPolicyCFG) -> None:
        if cfg.implementation is None:
            raise ValueError("ManaPolicyCFG.implementation is required")
        if cfg.embodiment is None:
            raise ValueError("ManaPolicyCFG.embodiment is required")
        if cfg.implementation.class_type is None:
            raise ValueError("ManaPolicyCFG.implementation.class_type is required")
        if not cfg.embodiment.action_mode:
            raise ValueError("ManaPolicyCFG.embodiment.action_mode is required")
        if int(cfg.embodiment.action_dim) <= 0:
            raise ValueError(
                f"ManaPolicyCFG.embodiment.action_dim must be > 0; "
                f"got {cfg.embodiment.action_dim} (set it in embodiments/<impl>/<EmbodimentClassName>.yaml)"
            )
        self.cfg = cfg
        self.embodiment = cfg.embodiment
        self.implementation = cfg.implementation.class_type()
        self.implementation.load(cfg.implementation)
        self.implementation.bind(self.embodiment)

        if cfg.remote is not None:
            if cfg.remote.class_type is None:
                raise ValueError("ManaPolicyCFG.remote.class_type is required when remote is set")
            self.remote = cfg.remote.class_type()
            self.remote.load(cfg.remote)
            self._get_actions_fn = self.remote.get_actions
        else:
            self._validate_local_inference_capability()
            self._get_actions_fn = self.implementation.get_actions

        self._client = cfg.client if cfg.client is not None else PolicyClientCFG()
        self._refill_policy = self._create_plugin(
            self._client.chunk_executor or PerEnvIndependentCFG(), kind="chunk_executor"
        )
        self._wire_hooks = [
            self._create_plugin(hook_cfg, kind="wire_interceptor") for hook_cfg in self._client.wire_interceptors
        ]
        self._postproc_fns = [
            self._create_plugin(pp_cfg, kind="chunk_postprocessor") for pp_cfg in self._client.chunk_postprocessors
        ]

        self._buffer = ChunkBuffer()
        self._chunk_scheduler = self._create_chunk_scheduler()
        self._is_loaded = True

    @staticmethod
    def _create_plugin(plugin_cfg: Any, *, kind: str) -> Any:
        """Instantiate a client plugin from its CFG via class_type + load."""
        plugin_class = getattr(plugin_cfg, "class_type", None)
        if not isinstance(plugin_class, type):
            raise ValueError(f"PolicyClientCFG.{kind}: {type(plugin_cfg).__name__}.class_type is required")
        plugin = plugin_class()
        plugin.load(plugin_cfg)
        return plugin

    def _validate_local_inference_capability(self) -> None:
        """Hard-fail at load time when no remote is attached and the impl
        does not override :meth:`PolicyImplementationBase.get_actions`.

        Lifting this check from first inference to load time matches PLAN.md
        §10.5 hard constraint #4 — every Policy yaml is provably runnable
        the moment ``ManaPolicy.load`` returns.
        """
        impl_cls = type(self.implementation)
        if impl_cls.get_actions is PolicyImplementationBase.get_actions:
            policy_name = self.cfg.name if self.cfg is not None else ""
            raise ValueError(
                f"Policy '{policy_name or impl_cls.__name__}' has no remote serving config and "
                f"{impl_cls.__name__} does not override get_actions(request); "
                "either set a remote CFG section in the policy yaml or use a local-inference "
                "implementation (gr00t / dp_image / replay)."
            )

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_actions(self, env: Any, observation: dict[str, Any]) -> torch.Tensor:
        """Single inference step over the chunk pipeline.

        Pipeline (PLAN.md §10.5):
          1. Delegate refill timing to ``client.chunk_scheduler``.
          2. Let the scheduler call ``_run_refill`` when it needs chunks.
          3. Write returned chunks into the buffer through ``_fill_chunks``.
          4. Pop one row per env and return ``(num_envs, action_dim)``.
        """
        if (
            not self._is_loaded
            or self.implementation is None
            or self.embodiment is None
            or self._get_actions_fn is None
        ):
            raise RuntimeError("ManaPolicy.load() must be called before get_actions()")

        action_dim = int(self.embodiment.action_dim)
        num_envs = int(getattr(env, "num_envs", 1))
        device = getattr(env, "device", "cpu")

        if self._chunk_scheduler is None:
            self._chunk_scheduler = self._create_chunk_scheduler()

        self.implementation.observe(observation)
        return self._chunk_scheduler.get_actions(
            env=env,
            observation=observation,
            num_envs=num_envs,
            action_dim=action_dim,
            device=device,
            buffer=self._buffer,
            client=self._client,
            run_refill=self._run_refill,
            fill_chunks=lambda chunks: self._fill_chunks(chunks),
        )

    def _run_refill(
        self,
        env: Any,
        observation: dict[str, Any],
        env_indices: list[int],
    ) -> dict[int, torch.Tensor | None]:
        refill_policy = self._refill_policy
        assert refill_policy is not None
        assert self.implementation is not None
        assert self._get_actions_fn is not None
        context_by_env = observation.get(REFILL_CONTEXT_KEY, {})
        if context_by_env:
            clean_observation = dict(observation)
            clean_observation.pop(REFILL_CONTEXT_KEY, None)
        else:
            clean_observation = observation

        def encode_request_with_context(obs: dict[str, Any], env_idx: int | None = None):
            request = self.implementation.encode_request(obs, env_idx=env_idx)
            if env_idx is not None and isinstance(context_by_env, dict):
                context = context_by_env.get(int(env_idx))
                if context is not None:
                    request[REFILL_CONTEXT_KEY] = context
            return request

        # websockets.sync clients are not re-entrant: background prefetch and
        # synchronous fallback must never call recv() on the same connection at
        # the same time.
        with self._inference_lock:
            return refill_policy(
                env=env,
                observation=clean_observation,
                env_indices=env_indices,
                encode_request=encode_request_with_context,
                inference_fn=self._get_actions_fn,
                decode_response=self.implementation.decode_response,
                wire_hooks=self._wire_hooks,
                postprocs=self._postproc_fns,
            )

    def _create_chunk_scheduler(self) -> ChunkSchedulerBase:
        scheduler_cfg = self._client.chunk_scheduler or SyncChunkSchedulerCFG()
        if scheduler_cfg.class_type is None:
            raise ValueError("PolicyClientCFG.chunk_scheduler.class_type is required")
        scheduler = scheduler_cfg.class_type()
        scheduler.load(scheduler_cfg)
        return scheduler

    def _fill_chunks(
        self,
        new_chunks: dict[int, torch.Tensor | None],
    ) -> None:
        for env_idx, chunk in new_chunks.items():
            if chunk is None:
                continue
            env_i = int(env_idx)
            prepared = self._trim_execute_horizon(chunk)
            if prepared.ndim >= 1 and int(prepared.shape[0]) > 0:
                connected = self._connect_chunk_if_enabled(prepared, env_idx=env_i)
                self._buffer.fill(env_i, connected)

    def _connect_chunk_if_enabled(
        self,
        chunk: torch.Tensor,
        *,
        env_idx: int,
    ) -> torch.Tensor:
        if not self._client.connect_chunks or self.embodiment is None:
            return chunk
        if chunk.ndim != 2:
            return chunk
        device = chunk.device
        return connect_action_chunk(
            chunk,
            last_command=self._buffer.get_last_command(env_idx),
            slot_layout=self.embodiment.slot_layout,
            ee_frames=self.embodiment.ee_frames,
            device=device,
        )

    def _trim_execute_horizon(self, chunk: torch.Tensor) -> torch.Tensor:
        """Keep only the first ``execute_horizon`` rows for execution (RTC)."""
        horizon = max(1, int(self._client.execute_horizon))
        if chunk.ndim == 2:
            return chunk[: min(int(chunk.shape[0]), horizon)]
        if chunk.ndim == 3:
            return chunk[:, : min(int(chunk.shape[1]), horizon)]
        return chunk

    def reset(
        self,
        env_ids: Any = None,
        *,
        episode_outcomes: dict[int, bool] | None = None,
    ) -> None:
        if self._chunk_scheduler is not None:
            self._chunk_scheduler.reset(env_ids)
        self._buffer.clear(env_ids)
        for hook in self._wire_hooks:
            hook.reset(env_ids)
        if self.implementation is not None:
            finalize_episode = getattr(self.implementation, "finalize_episode", None)
            if callable(finalize_episode):
                finalize_episode(env_ids, episode_outcomes)
            initialize_episode = getattr(self.implementation, "initialize_episode", None)
            if callable(initialize_episode):
                initialize_episode(env_ids)
            else:
                # Third-party implementations written before the lifecycle
                # contract may still expose only the generic local reset hook.
                self.implementation.reset(env_ids)
        if self.remote is not None:
            self.remote.reset(env_ids)

    def cleanup(self) -> None:
        if self._chunk_scheduler is not None:
            self._chunk_scheduler.cleanup()
        if self.implementation is not None:
            self.implementation.cleanup()
        if self.remote is not None:
            self.remote.cleanup()

    def requires_serial_envs(self) -> bool:
        """Whether the active implementation owns one shared episode state."""
        return bool(self.implementation and self.implementation.requires_serial_envs())

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


@dataclass
class ManaPolicyCFG:
    """CFG for :class:`ManaPolicy`. Holds three orthogonal sub-CFG
    references plus the client-side segment (see PLAN.md §10.4 + §10.5).

    Required:
      * ``implementation`` - PolicyImplementationCFG instance
      * ``embodiment``     - PolicyEmbodimentCFG instance (injected by
                              DataCollectionFactory after the runtime
                              embodiment Python class is known)

    Optional:
      * ``remote``         - PolicyRemoteCFG instance, or ``None``. When
                              present the Policy binds ``remote.get_actions``
                              as its inference function; when absent the
                              impl must override :meth:`get_actions`
                              (validated by :meth:`ManaPolicy._validate_local_inference_capability`
                              at load time).
      * ``client``         - PolicyClientCFG instance with the resolved
                              chunk pipeline. ``None`` -> default
                              (execute_horizon=1, no hooks / postprocs).
    """

    name: str = ""
    implementation: PolicyImplementationCFG | None = None
    remote: PolicyRemoteCFG | None = None
    embodiment: PolicyEmbodimentCFG | None = None
    client: PolicyClientCFG | None = None
    # Action interface this policy drives. Empty => use the embodiment default
    # (PolicyEmbodimentCFG.action_mode). When set (e.g. ``pink_ik`` for the pi0
    # an OpenPI policy whose embodiment default is ``ik_eef_abs``) it selects
    # one of the embodiment's available modes. The policy owns "which mode I
    # want"; the embodiment owns "available modes + default". Replaces the former
    # order-level ``BenchmarkOrderCFG.action_mode`` / ``DataCollectionOrderCFG.action_mode``.
    action_mode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    class_type: type = ManaPolicy


def create_policy(cfg: ManaPolicyCFG) -> ManaPolicy:
    """Instantiate a :class:`ManaPolicy` from *cfg* and load it."""
    policy_class = cfg.class_type
    if policy_class is None:
        raise ValueError("ManaPolicyCFG.class_type is required to create a policy")
    policy = policy_class()
    policy.load(cfg)
    return policy
