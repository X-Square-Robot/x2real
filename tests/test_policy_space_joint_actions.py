from dataclasses import replace

import numpy as np
import pytest

from manaenv.policy.action_mode_selection import clone_embodiment_for_action_mode
from manaenv.policy.mana_policy import ChunkBuffer
from manaenv.policy.policy_embodiment import load_policy_embodiment_cfg
from manaenv.policy.policy_implementation.policy_space import PolicySpaceImpl, PolicySpaceImplCFG


@pytest.mark.parametrize(
    ("schema", "wire_dim", "sim_dim"),
    [
        ("bimanual_joint_position@1", 14, 14),
        ("bimanual_joint_position_26d@1", 26, 14),
        ("bimanual_joint_position_26d@1", 26, 26),
        ("", 26, 14),
    ],
)
def test_joint_wire_fits_configured_action_buffer(schema, wire_dim, sim_dim):
    embodiment = load_policy_embodiment_cfg(
        implementation_yaml_name="policy_space", embodiment_class_name="EX0016REmbodiment"
    )
    embodiment = clone_embodiment_for_action_mode(embodiment, "joint_position")
    embodiment = replace(embodiment, action_dim=sim_dim)
    implementation = PolicySpaceImpl()
    implementation.load(
        PolicySpaceImplCFG(
            action_type="absolute_joint",
            observation_schema="bimanual_rgb_joint@1" if schema else "",
            action_schema=schema,
            embodiment_schema="ex001_6r@1" if schema else "",
        )
    )
    implementation.bind(embodiment)
    wire = np.arange(3 * wire_dim, dtype=np.float32).reshape(3, wire_dim)

    chunk = implementation.decode_response({"actions": wire}, env_idx=0)
    buffer = ChunkBuffer()
    buffer.fill(0, chunk)
    action = buffer.pop_step(num_envs=1, action_dim=sim_dim, device="cpu")

    assert tuple(chunk.shape) == (3, sim_dim)
    np.testing.assert_array_equal(chunk.numpy(), wire[:, :sim_dim])
    np.testing.assert_array_equal(action.numpy(), wire[:1, :sim_dim])


def test_joint_schema_is_validated_before_trimming_recording_channels():
    implementation = PolicySpaceImpl()
    implementation.load(
        PolicySpaceImplCFG(
            action_type="absolute_joint",
            observation_schema="bimanual_rgb_joint@1",
            action_schema="bimanual_joint_position@1",
            embodiment_schema="ex001_6r@1",
        )
    )
    embodiment = load_policy_embodiment_cfg(
        implementation_yaml_name="policy_space", embodiment_class_name="EX0016REmbodiment"
    )
    implementation.bind(clone_embodiment_for_action_mode(embodiment, "joint_position"))

    with pytest.raises(RuntimeError, match="requires width 14, got 26"):
        implementation.decode_response({"actions": np.zeros((3, 26), dtype=np.float32)})
