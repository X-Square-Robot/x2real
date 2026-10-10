# 原子能力菜单 (SKILLS)

> 自动生成,勿手改。源是 `manaenv/terms/terminations.py` 各谓词 docstring 的 `[SKILL <cat>]` + `Assumes:` 标签;
> 改标签后重跑 `python manaenv/scripts/gen_skills_catalog.py`。一种能力多个谓词 = 合法变体(假设不同),
> 选哪个由 agent/用户拿自己的机器人+环境去匹配 `Assumes`,库不替你选。

共 **57** 个能力谓词,**10** 类。

## reach

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `move_target` | EE tip within ``max_distance`` of a fixed env-frame target point. | one EE FrameTransformer frame (target_pos_w/target_quat_w); optional tip_offset in EE local frame; object-agnostic (caller computes target_pos upstream). | 2311 |
| `ee_near_init_pose` | All EE frames within ``max_distance`` of their episode-init pose (+offset); ANDs across arms. | EE FrameTransformer frames (target_pos_w); capture_init_ee_poses event ran at reset (env._mana_init_ee_pos_w). | 2348 |
| `ee_near_object` | ANY EE frame within ``max_distance`` of the target object (+offset); ORs across arms. | EE FrameTransformer frames; object pool pose (root_pos_w); respects per-env pinned grasp target (EPISODE_GRASP_TARGET_ATTR) when set. | 2392 |
| `robot_body_near_object_xy` | A robot body (e.g. EE) within ``xy_threshold`` m of an object on the XY plane (Z ignored). | robot articulation with named body (body_pos_w); object pose (root_pos_w); OR across an object list. | 4198 |

## grasp

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `gripper_grasps_object` | Gripper contact force above ``force_threshold`` AND object lifted above ``lifted_z_min``. | gripper contact sensor (net_forces_w); object pose (root_pos_w); only first object of a list checked; default lifted_z_min calibrated for the EX001-6R apple scene. | 2086 |
| `grasp_left_1` | Per-arm subtask signal: left arm grasps and lifts ``object_cfg`` (wraps gripper_grasps_object on left_gripper_contact). | EX001-6R ``left_gripper_contact`` sensor pre-registered on the embodiment; consumed by name as a mimic subtask-term signal. | 2153 |
| `grasp_right_1` | Per-arm subtask signal: right arm grasps and lifts ``object_cfg`` (wraps gripper_grasps_object on right_gripper_contact). | EX001-6R ``right_gripper_contact`` sensor pre-registered on the embodiment; consumed by name as a mimic subtask-term signal. | 2178 |
| `gripper_close_and_contact` | Per-arm (gripper closed AND contact), combined across arms (all/any). | robot articulation with named gripper joint(s) (joint_pos); per-arm contact sensor(s) (net_forces_w); single-arm scalar or bimanual paired lists. | 2468 |

## lift

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `object_above_height` | Object env-frame z-position >= ``min_height`` (raw absolute height, no anchor). | object pose (root_pos_w). | 2201 |
| `object_lifted` | Object env-z risen >= ``min_lift`` m above an anchor (settle-time or first-contact). | object pose (root_pos_w); episode_length_buf; optional contact sensor(s) (net_forces_w) for grasp-moment anchoring. | 2601 |
| `indexed_object_lifted` | Lift check on the single object selected by a reset-committed index. | ``index_log_key`` stores one candidate-aligned integer per env and candidate root poses are available. | 2687 |
| `grasp_target_lifted` | The episode's pinned grasp-target object is contacted AND lifted (lift implies grasp). | per-object contact sensors by name convention ``{object}_{sensor_suffix}``; per-env pin (EPISODE_GRASP_TARGET_ATTR); object pose; pool+pin model mirrors ee_near_object. | 2735 |
| `cup_reveals_hidden_object` | Detect a lifted target cup that no longer covers the hidden object. | cup and hidden-object root poses are available in the active scene pool. | 4751 |

## place

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `object_position_in_zone` | Object position inside a configured zone (abs box or relative-to-reference). | object pose (root_pos_w); optional reference entity; optional velocity gate (root_lin_vel_w); consecutive-steps support. | 675 |
| `objects_outside_zone` | Every listed object lies outside a configured zone. |  | 854 |
| `object_in_opened_joint_region` | Object XY inside the live opened-joint placeable tray region. |  | 1243 |
| `indexed_object_position_in_zone` | Zone check on the single object selected by a reset-committed index. | ``index_log_key`` stores one candidate-aligned integer per env and object poses are available. | 1323 |
| `object_near_nearest_reference_child_xy` | Object root XY within ``xy_distance_threshold`` of a dock child slot. |  | 1662 |
| `objects_fraction_in_relative_zone` | At least ``min_fraction`` of objects lie in a relative zone of ``reference_cfg``. | the reference and candidate objects expose root poses; optional velocity checks need root velocity. | 4546 |
| `objects_count_in_relative_zone` | At least ``min_count`` objects lie in a relative zone of ``reference_cfg``. |  | 4604 |
| `objects_cleared_from_relative_zone` | Detect when no listed object remains in a reference-relative zone. | the reference and candidate objects expose root poses; optional velocity checks need root velocity. | 4660 |
| `cup_covers_hidden_object` | Detect an inverted cup covering a hidden object. | cup pose and velocity plus hidden-object pose are available in the active scene pool. | 4698 |
| `attribute_targets_in_basket` | Every predicate-matching object rests in the basket (strict: non-matching stay out). |  | 4939 |
| `any_object_in_zone` | Any of a candidate object set rests inside a fixed env-frame box. |  | 5157 |
| `any_object_in_zone_for_duration` | Active object continuously inside a box for >= ``hold_seconds`` (dwell). |  | 5184 |
| `any_object_in_indexed_zone` | Active object inside the box selected per-env by a reset-committed index. |  | 5218 |
| `any_object_in_indexed_zone_for_duration` | Indexed-zone dwell for >= ``hold_seconds`` (reset when object leaves). | ``index_log_key`` stores one zone-aligned integer per env and runtime step duration is available. | 5266 |
| `classified_object_on_color_target` | Reset-selected object rests on its color-matched destination. | reset commits 1-based object/color indices; runtime asset slots expose root pose and velocity. | 5648 |

## insert

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `object_inserted` | Object inserted into target along an axis (axial depth + lateral offset + orientation aligned). | object + target pose (root_pos_w) and orientation (root_quat_w, xyzw) + object velocity; optional insertion_axis (default target local Z) and object_axis (default object local Z). | 3067 |
| `usb_inserted_at_target_port` | USB seated in the sampled hub port with correct face; hub unmoved. |  | 6979 |
| `shape_piece_in_target_slot` | Shape piece seated in its slot marker; sorter base unmoved. |  | 7107 |

## orient

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `object_orientation_near_default` | Live root quat stays near the scene/default spawn orientation. |  | 1745 |
| `object_tilt_in_range` | Object local ``axis`` alignment with world up (dot) within [*min_dot*, *max_dot*] (e.g. upright/flat). | object orientation (root_quat_w, xyzw) + velocity (root_lin_vel_w); list ANDs across all. | 2832 |
| `object_euler_xy_out_of_range` | Object roll or pitch OUTSIDE ``[min_deg, max_deg]`` (negative-sense: tumble/fallen detection). | object orientation (root_quat_w, xyzw) + optional velocity gate; list ORs across any. | 2921 |
| `object_yaw_in_range` | \|yaw\| of object within [*min_yaw_deg*, *max_yaw_deg*] (or any of *yaw_ranges*). | object orientation (root_quat_w, xyzw) + velocity (root_lin_vel_w). | 2959 |

## contact

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `object_contact_destination` | Filtered contact force between pickable and destination stays ABOVE threshold. | one or more contact sensors with filter_prim_paths to the destination (force_matrix_w); consecutive-steps. | 2026 |
| `object_no_contact_destination` | Filtered contact force between pickable and destination stays BELOW threshold (inverse). | one or more filtered contact sensors on the pickable (force_matrix_w); consecutive-steps. | 2055 |
| `contact_once` | Latches True once a single contact sensor fires above ``force_threshold`` (per episode). | one contact sensor (force_matrix_w); episode_length_buf/reset_buf for the latch; dual-mode (eval vs termination). | 3163 |
| `contact_count_at_least` | At least ``min_count`` of the given contact sensors are active (latched unless consecutive_steps>1). | multiple contact sensors (force_matrix_w); latches flags at episode reset. | 3204 |
| `contact_sequence_reached` | Ordered contact sequence reached up to ``sequence_length`` (live progress). | multiple contact sensors; order via sequence_order/button_order or parsed from instruction (env._mana_benchmark_instruction). | 3244 |
| `contact_sequence_by_first_touch` | Ordered contact sequence by first-touch timestamp (ascending first-touch steps), not live contact. | multiple contact sensors; same ordering modes as contact_sequence_reached; tracks first-touch step per sensor. | 3326 |

## joint

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `articulation_joints_near_default` | Current joint positions within ``atol`` rad of the articulation default (USD/reset) pose. | articulation joints (joint_pos/default_joint_pos); optional joint_names subset (whole-arm homing if omitted). | 3388 |
| `articulation_joints_near_default_after_time` | Joints within ``atol`` rad of default pose AND elapsed time > ``after_seconds`` (homed after a window). | articulation joints (joint_pos/default_joint_pos); a runtime with step_dt_s + frame counter (or env.episode_length_buf). | 3437 |
| `joint_above_threshold` | Per-joint normalized openness vs ``threshold`` (all/any), Arena ``Openable``-compatible. | articulation with joint_pos + joint_pos_limits; joint_names are regex (fullmatch); consecutive-steps support. | 3578 |
| `indexed_joint_above_threshold` | Openness check on the single joint selected by a reset-committed index. | the articulation exposes all ordered joints and ``index_log_key`` stores a matching integer per env. | 3629 |
| `joint_reaches_sampled_openness` | Success when the knob reaches the task-pinned target openness. |  | 3687 |
| `joint_above_threshold_once` | Latching ``joint_above_threshold`` — True once satisfied at any point in the episode. | same as joint_above_threshold; latch resets on episode reset. | 3822 |
| `pump_pressed_n_times` | Count pump press-release cycles until the sampled target is reached. | one articulated pump joint has calibrated limits and ``count_log_key`` stores the target per env. | 3876 |
| `knob_at_target_level` | Turnable knob is at the reset-sampled target temperature level. |  | 3999 |

## multi-object

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `any_object_displaced_from_anchor` | Detect any listed object displaced from its stage anchor. | every listed object exposes a root pose and the term is evaluated once before displacement. | 2553 |
| `objects_equidistant` | Consecutive objects equidistantly spaced (max−min spacing < tolerance). | >=3 objects with pose (root_pos_w) + velocity (root_lin_vel_w). | 3019 |
| `gripper_nth_object_lifted` | The *nth* gripper-touched object (by touch order) has been lifted >= ``min_lift``. | parallel lists gripper_contact_sensor_cfgs + object_cfgs (net_forces_w / root_pos_w); lift vs object env-z at episode start. | 4304 |
| `gripper_nth_object_settled` | The *nth* gripper-touched object is back near the table and released. | parallel contact sensors and object configs expose forces, root poses, and linear velocities. | 4351 |
| `nth_object_near_mth_object` | The *nth* gripper-touched object within ``xy_distance_threshold`` (XY) of the *mth* touched object. | parallel gripper_contact_sensor_cfgs + object_cfgs; object pose (root_pos_w); identities resolved by touch order. | 4416 |
| `nth_object_contact_others` | The *nth* gripper-touched object has inter-object contact force above threshold. | parallel gripper_contact_sensor_cfgs + inter_object_contact_sensor_cfgs (force_matrix_w); identity by touch order. | 4453 |
| `cups_swapped_n_times` | Count settled position swaps between two cups. | both cup poses and velocities exist and ``count_log_key`` stores the target per env. | 4794 |

## navigation

| 谓词 | 检测什么 | 机器人/环境假设 (Assumes) | 行号 |
|---|---|---|---|
| `base_in_operation_zone` | Mobile base inside a compiled operation zone with speed gates. | EX001 root_pos_w/quat_w(XYZW)/lin_vel_b/ang_vel_b; zone bounds from runtime_context (no YAML IO). | 7209 |
