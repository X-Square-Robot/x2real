"""Franka Emika Panda embodiment for manaenv.

Thin manaenv-side wrapper around IsaacLab-Arena's :class:`FrankaIKEmbodiment`.
The arena class already supplies the full scene / observation / action /
event wiring for a single-arm Panda with a parallel-jaw hand
(``panda_joint1..7`` + ``panda_finger_joint1/2``, EE link ``panda_hand``).
This module only:

1. Re-registers the embodiment under the manaenv asset name ``"franka"`` so
   :class:`manaenv.pipeline.factories.robot_factory.RobotFactory` can resolve
   it via ``ModuleImportSpec.from_symbol("manaenv.assets.robots", "franka")``
   (``gen_method=0``, i.e. YAML ``robot.definition.tags: franka``).
2. Repoints the spawn USD at the public Isaac asset
   ``Robots/FrankaRobotics/FrankaPanda/franka.usd`` (Panda arm + parallel hand
   in one file) instead of arena's internal ``franka_panda_hand_on_stand.usd``.
   Override with ``MANAENV_FRANKA_USD_PATH`` (set to ``instanceable`` to use the
   lighter ``FrankaEmika/panda_instanceable.usd``, or to any explicit URL/path).
3. Attaches the manaenv :class:`FrankaCameraCfg` (wrist + top) so the
   embodiment carries the public evaluation camera bundle directly.
4. Keeps the public Franka path on Arena's PhysX differential-IK action.

The USD ships its parallel-jaw gripper under the ``Gripper=Default`` variant;
``franka.usd`` selects that variant by default, so no ``spawn.variants``
override is needed (that is only required for the Robotiq variant).
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import torch
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment

from manaenv.actuators.select import apply_actuator
from manaenv.assets.resolver import resolve_runtime_asset_reference
from manaenv.assets.sensors.franka_cameras import FrankaCameraCfg
from manaenv.utils.debug_print import manaprint

if TYPE_CHECKING:
    from isaaclab_arena.utils.pose import Pose


def _franka_arm_joint_pos(env, asset_cfg=None, joint_prefix: str = "panda") -> torch.Tensor:
    """Absolute 7-D Panda arm joints for PolicySpace ``absolute_joint``."""
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    robot = env.scene[asset_cfg.name]
    joint_names = [f"{joint_prefix}_joint{i}" for i in range(1, 8)]
    name_to_idx = {name: i for i, name in enumerate(robot.data.joint_names)}
    missing = [name for name in joint_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Franka arm_joint_pos missing joints {missing}; have {list(robot.data.joint_names)}")
    indices = [name_to_idx[name] for name in joint_names]
    return (robot.data.joint_pos).torch[:, indices]


def _franka_gripper_pos_norm(env, asset_cfg=None, finger_joint_name: str = "panda_finger_joint1") -> torch.Tensor:
    """1-D gripper state in ``[0, 1]`` (0=open, 1=closed) for PolicySpace."""
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    robot = env.scene[asset_cfg.name]
    try:
        idx = robot.data.joint_names.index(finger_joint_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Franka gripper_pos missing {finger_joint_name}; have {list(robot.data.joint_names)}"
        ) from exc
    # Parallel-jaw open width ~0.04 m; close -> 0.0.
    open_width = 0.04
    finger = (robot.data.joint_pos).torch[:, idx : idx + 1]
    closed = 1.0 - (finger / open_width)
    return torch.clamp(closed, 0.0, 1.0)


# Public Isaac asset: Panda arm + parallel hand in a single USD. Same path
# string IsaacLab itself uses for ``FRANKA_ROBOTIQ_GRIPPER_CFG`` (default
# ``Gripper=Default`` variant gives the standard panda parallel-jaw hand).
_FRANKA_PANDA_HAND_USD = "assets/robots/franka/franka.usd"

# Lighter instanceable Panda (IsaacLab's FRANKA_PANDA_CFG default). Same joint
# / link names, so it is a drop-in alternative for spawn-only validation. Lives
# within the user-supplied local robot asset tree.
_FRANKA_INSTANCEABLE_USD = "assets/robots/franka/panda_instanceable.usd"


def _franka_usd_path() -> str:
    """Resolve the Franka spawn USD.

    Default is the public ``FrankaPanda/franka.usd`` (Panda + parallel hand).
    Override with ``MANAENV_FRANKA_USD_PATH``:

    - ``instanceable`` -> ``FrankaEmika/panda_instanceable.usd``
    - any other non-empty value -> used verbatim as the USD path/URL
    """
    override = os.environ.get("MANAENV_FRANKA_USD_PATH", "").strip()
    if not override:
        return resolve_runtime_asset_reference(_FRANKA_PANDA_HAND_USD)
    if override.lower() == "instanceable":
        return resolve_runtime_asset_reference(_FRANKA_INSTANCEABLE_USD)
    return resolve_runtime_asset_reference(override)


@register_asset
class FrankaEmbodiment(FrankaIKEmbodiment):
    """Franka Panda (single 7-DOF arm + parallel-jaw hand), differential-IK arm control.

    Resolved by manaenv ``RobotFactory`` under the asset name ``"franka"``.
    Inherits scene / observation / action / event / reward config from
    :class:`isaaclab_arena.embodiments.franka.franka.FrankaIKEmbodiment`; only
    the spawn USD and the default camera bundle differ.
    """

    name = "franka"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        disable_gripper_shadow: bool = False,
        disable_scene_shadow: bool = False,
        actuator: str = "pd",
        actuator_params: dict | None = None,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            initial_joint_pose=initial_joint_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
        )
        # Repoint spawn USD at the public Panda+hand asset while preserving the
        # high-PD articulation/actuator tuning arena set on ``scene_config.robot``.
        self.scene_config.robot.spawn.usd_path = _franka_usd_path()
        # Select the actuator control law (default pd = keep arena's Implicit
        # cfgs untouched). No StablePD gain table here, so stablepd does a generic
        # Implicit->StablePD field copy.
        apply_actuator(self.scene_config.robot, actuator, actuator_params=actuator_params)
        # Adapt both the legacy and current Arena Franka reset schemas to one
        # deterministic direct-write event.
        self._install_manaenv_reset_event()
        # Carry the public wrist + top camera bundle directly.
        # RobotFactory's _CAMERA_CFG_REGISTRY fallback only fires when
        # camera_config is None, so set it explicitly.
        self.camera_config = FrankaCameraCfg()
        # Optionally stop hand/finger links (disable_gripper_shadow) OR the whole
        # cloned env subtree (disable_scene_shadow) from casting RTX shadows. Both
        # off by default so other scenes render unchanged. scene_wide supersedes
        # the gripper-only path (it already covers the gripper).
        self._maybe_disable_shadows(disable_gripper_shadow, disable_scene_shadow)
        # Mount finger contact for grasp success after the Panda config is ready.
        self._install_gripper_contact_sensor()
        # PolicySpace joint policies (dreamzero_droid / cosmos3_droid): absolute
        # arm joints + normalized 1-D gripper observations.
        self._install_policy_space_observations()
        # Arena sets ``self.JointActionsCfg = FrankaJointActionsCfg`` (sign-channel
        # gripper). PolicySpace absolute_joint uses [0,1] gripper; replace the
        # instance pointer with a zero-arg factory so action_mode selection works.
        self.JointActionsCfg = self._policy_space_joint_actions_cfg

    def _finger_name_prefix(self) -> str:
        """Return the fixed Panda joint/body prefix."""
        return "panda"

    def _install_gripper_contact_sensor(self) -> None:
        """Attach a finger ContactSensor used by nut_on_screw / grasp success."""
        from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
        from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

        self.scene_config.robot.spawn.activate_contact_sensors = True
        prefix = self._finger_name_prefix()
        sensor_cfg = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{prefix}_.*finger",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
        )
        contact_scene_cfg = make_configclass(
            "FrankaGripperContactCfg", [("gripper_contact", ContactSensorCfg, sensor_cfg)]
        )()
        self.scene_config = combine_configclass_instances(
            "FrankaContactSceneCfg", self.scene_config, contact_scene_cfg
        )

    def get_ee_contact_sensor_names(self, arm_mode: str | None = None) -> list[str]:
        del arm_mode
        return ["gripper_contact"]

    def get_ee_contact_prim_paths(self, arm_mode: str | None = None) -> list[str]:
        del arm_mode
        prefix = self._finger_name_prefix()
        return [
            f"{{ENV_REGEX_NS}}/Robot/{prefix}_leftfinger",
            f"{{ENV_REGEX_NS}}/Robot/{prefix}_rightfinger",
        ]

    def get_gripper_states(self) -> dict[str, float]:
        """Map robot-agnostic gripper symbols to this robot's action-term values.

        The default binary gripper action is a SIGN channel (float ``>=0`` ->
        open, ``<0`` -> close; the actual finger widths live in the action's
        ``open/close_command_expr``), so symbols map to +/-1.
        """
        return {"open": 1.0, "closed": -1.0}

    def ActionsCfg(self):  # noqa: N802 - looked up by name as an ActionsCfg factory
        """Planner ``action_mode='ik_eef_delta'`` factory hook.

        ``apply_planner_requested_action_mode`` calls ``embodiment.ActionsCfg()``
        and assigns the result to ``action_config``. Franka's default action
        config already IS the relative diff-IK cfg (arena ``FrankaIKActionCfg``,
        ``use_relative_mode=True``) with the ctor customizations applied in
        place (the manaenv reset/camera customizations) -- return it as-is
        instead of minting a fresh arena cfg, which would silently drop those
        tweaks.
        """
        return self.action_config

    def get_ee_frame_name(self, arm_mode: str | None = None) -> str:
        del arm_mode
        return "ee_frame"

    def get_ee_frame_names(self, arm_mode: str | None = None) -> list[str]:
        del arm_mode
        return ["ee_frame"]

    def get_command_body_name(self) -> str:
        return f"{self._finger_name_prefix()}_hand"

    def get_command_body_names(self, arm_mode: str | None = None) -> list[str]:
        del arm_mode
        body = self.get_command_body_name()
        return [body, body]

    def ActionsCfgAbsIK(self):  # noqa: N802 - looked up by name as an ActionsCfg factory
        """Policy ``action_mode='ik_eef_abs'`` factory hook.

        Left arm: absolute DiffIK + raw finger driver. Right-arm slots are
        no-op channels so the shared 16-D policy wire stays dual-arm shaped.
        PolicySpace benchmarks use ``JointActionsCfg`` (joint-8) instead.
        """
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
        from isaaclab.managers import ActionTermCfg
        from isaaclab.utils.configclass import configclass

        from manaenv.actions.clamped_raw_gripper import ClampedRawGripperActionCfg
        from manaenv.actions.command_rate_diff_ik import CommandRateDifferentialInverseKinematicsActionCfg
        from manaenv.actions.noop_action import NoopActionCfg

        ac = self.action_config
        arm = getattr(ac, "arm_action", None)
        prefix = self._finger_name_prefix()
        joint_names = list(getattr(arm, "joint_names", None) or [f"{prefix}_joint.*"])
        body_name = str(getattr(arm, "body_name", None) or f"{prefix}_hand")
        body_offset = getattr(arm, "body_offset", None)
        offset_pos = [0.0, 0.0, 0.1034]
        if body_offset is not None and getattr(body_offset, "pos", None) is not None:
            offset_pos = [float(v) for v in body_offset.pos]

        @configclass
        class _FrankaActionsCfgAbsIK:
            arm_action: ActionTermCfg = CommandRateDifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=joint_names,
                body_name=body_name,
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                scale=1.0,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=offset_pos),
            )
            gripper_action: ActionTermCfg = ClampedRawGripperActionCfg(
                asset_name="robot",
                joint_names=[f"{prefix}_finger_joint1"],
                scale=1.0,
                offset=0.0,
                use_default_offset=False,
                clamp_min=0.0,
                clamp_max=0.04,
            )
            right_arm_action: ActionTermCfg = NoopActionCfg(asset_name="robot", action_dim=7)
            right_gripper_action: ActionTermCfg = NoopActionCfg(asset_name="robot", action_dim=1)

        return _FrankaActionsCfgAbsIK()

    def _policy_space_joint_actions_cfg(self):
        """Absolute joint-8 config for PolicySpace ``action_mode='joint_position'``.

        Matches DROID / dreamzero_droid: 7 arm joints + gripper in ``[0, 1]``
        (0=open, 1=closed via ZeroToOne threshold at 0.5).
        """
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
        from isaaclab.managers import ActionTermCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_arena.embodiments.droid.droid import BinaryJointPositionZeroToOneActionCfg

        prefix = self._finger_name_prefix()
        open_w = 0.04
        close_w = 0.0
        finger_expr = f"{prefix}_finger_.*"

        @configclass
        class _FrankaPolicySpaceJointActionsCfg:
            body: ActionTermCfg = JointPositionActionCfg(
                asset_name="robot",
                joint_names=[f"{prefix}_joint.*"],
                preserve_order=True,
                use_default_offset=False,
            )
            gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
                asset_name="robot",
                joint_names=[finger_expr],
                open_command_expr={finger_expr: open_w},
                close_command_expr={finger_expr: close_w},
            )

        return _FrankaPolicySpaceJointActionsCfg()

    def _install_policy_space_observations(self) -> None:
        """Install PolicySpace joint obs: ``arm_joint_pos`` + 1-D ``gripper_pos``."""
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        policy = getattr(self.observation_config, "policy", None)
        if policy is None:
            return
        prefix = self._finger_name_prefix()
        policy.arm_joint_pos = ObsTerm(
            func=_franka_arm_joint_pos,
            params={"joint_prefix": prefix},
        )
        # Arena Franka ``gripper_pos`` is 2-D finger widths; PolicySpace wants 1-D.
        policy.gripper_pos = ObsTerm(
            func=_franka_gripper_pos_norm,
            params={"finger_joint_name": f"{prefix}_finger_joint1"},
        )
        manaprint(f"INFO: Franka PolicySpace ObsTerms installed " f"(arm_joint_pos + gripper_pos; prefix={prefix})")

    def _install_manaenv_reset_event(self) -> None:
        """Adapt Arena's Franka reset schema to one direct-write reset event.

        Older Arena revisions expose ``init_franka_arm_pose`` plus a gaussian
        writer. Newer revisions own the initial pose on
        ``scene_config.robot.init_state.joint_pos`` and expose only the gaussian
        writer. Reusing a declared event slot keeps the term discoverable by
        Isaac Lab's config manager.
        """
        from isaaclab.managers import EventTermCfg as EventTerm

        from manaenv.terms.events import set_arm_joint_pose_to_sim

        ec = self.event_config
        if ec is None:
            return
        legacy_term = getattr(ec, "init_franka_arm_pose", None)
        default_pose = None
        if legacy_term is not None and isinstance(getattr(legacy_term, "params", None), dict):
            default_pose = legacy_term.params.get("default_pose")
        if default_pose is None:
            default_pose = self._ordered_scene_joint_pose()

        direct_term = EventTerm(
            func=set_arm_joint_pose_to_sim,
            mode="reset",
            params={"default_pose": default_pose},
        )

        if legacy_term is not None:
            ec.init_franka_arm_pose = direct_term
        elif hasattr(ec, "randomize_franka_joint_state"):
            ec.randomize_franka_joint_state = direct_term
        else:
            raise RuntimeError("Arena FrankaEventCfg exposes no reset-event slot")

        if legacy_term is not None and hasattr(ec, "randomize_franka_joint_state"):
            ec.randomize_franka_joint_state = None

    def _ordered_scene_joint_pose(self) -> list[float]:
        """Expand the scene joint-position patterns into Panda joint order."""
        robot_cfg = getattr(getattr(self, "scene_config", None), "robot", None)
        init_state = getattr(robot_cfg, "init_state", None)
        joint_pos = getattr(init_state, "joint_pos", None)
        if not isinstance(joint_pos, dict):
            raise RuntimeError("Arena Franka scene config has no joint_pos mapping")

        prefix = self._finger_name_prefix()
        joint_names = [f"{prefix}_joint{i}" for i in range(1, 8)]
        joint_names.extend([f"{prefix}_finger_joint1", f"{prefix}_finger_joint2"])
        ordered_pose: list[float] = []
        for joint_name in joint_names:
            matches = [value for pattern, value in joint_pos.items() if re.fullmatch(str(pattern), joint_name)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Arena Franka joint_pos must resolve {joint_name!r} exactly once; "
                    f"matched {len(matches)} pattern(s) from {list(joint_pos)}"
                )
            ordered_pose.append(float(matches[0]))
        return ordered_pose

    def _maybe_disable_shadows(
        self,
        disable_gripper_shadow: bool = False,
        disable_scene_shadow: bool = False,
    ) -> None:
        """Stop the gripper, or the whole scene, from casting RTX shadows.

        Two scopes, each gated by its own YAML flag, both off by default so every
        other scene renders unchanged:

        * ``disable_gripper_shadow`` -- only the hand/finger links stop casting;
          the table texture and the bricks' own shadows stay.
        * ``disable_scene_shadow`` -- the entire cloned env subtree (robot + table
          + board + bricks) stops casting, removing the hard drop-shadow the
          directional key light would otherwise throw onto the table. Supersedes
          the gripper-only scope (it already covers the gripper), so when both are
          on only the scene-wide event is installed.

        The brick-stacking scene lights with a directional ``DistantLight`` key
        (see ``stacking_blocks.py::BrickRtxKeyLight``). Being directional it casts
        a hard contact shadow under Kit's RTX renderer. The global
        ``/rtx/shadows/enabled`` carb switch can't remove it (no-op under Kit's
        path-tracing mode), so we author the per-prim primvar
        ``primvars:doNotCastShadows`` via a ``startup``-mode event -- Kit's renderer
        honours that primvar in both real-time and PT modes.

        ``disable_scene_shadow`` removes cast shadows but does NOT brighten the
        scene by itself; pair it with a key-light intensity /
        ``/rtx/sceneDb/ambientLightIntensity`` bump in the scene YAML if a higher
        overall level is wanted.

        The Panda hand/finger link names are passed for the gripper scope.
        """
        if not disable_gripper_shadow and not disable_scene_shadow:
            return

        ec = self.event_config
        if ec is None:
            return

        from isaaclab.managers import EventTermCfg as EventTerm

        from manaenv.terms.events import disable_cast_shadows

        if disable_scene_shadow:
            # Scene-wide already covers the gripper, so install just this one.
            ec.disable_gripper_shadow = EventTerm(
                func=disable_cast_shadows,
                mode="startup",
                params={"scene_wide": True},
            )
            manaprint(
                "INFO: Scene shadow casting disabled (primvars:doNotCastShadows on "
                "each cloned env root via startup event); all DomeLight contact "
                "shadows removed (robot + table + board + bricks)."
            )
            return

        # The event drops links whose prim does not exist, so this remains robust
        # across the Panda USD variants.
        body_names = [
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        ec.disable_gripper_shadow = EventTerm(
            func=disable_cast_shadows,
            mode="startup",
            params={"body_names": body_names},
        )
        manaprint(
            "INFO: Franka gripper shadow disabled (primvars:doNotCastShadows on "
            "hand/finger links via startup event); DomeLight contact shadow removed."
        )
