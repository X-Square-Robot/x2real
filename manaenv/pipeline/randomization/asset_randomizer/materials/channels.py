from __future__ import annotations


class MaterialChannel:
    BASE_COLOR = "base_color"
    METALLIC = "metallic"
    ROUGHNESS = "roughness"
    SPECULAR = "specular"
    OPACITY = "opacity"
    NORMAL = "normal"
    ORM = "orm"


class ChannelSource:
    CONSTANT = "constant"
    TEXTURE = "texture"
    PACKED_TEXTURE = "packed_texture"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


class RandomizationOp:
    SET_VALUE = "set_value"
    CHOICE = "choice"
    SAMPLE_UNIFORM = "sample_uniform"
    HSV_JITTER = "hsv_jitter"
    JITTER_RELATIVE = "jitter_relative"
    CLAMP = "clamp"

    SKIP = "skip"
    REPLACE = "replace"
    SCALE = "scale"
    BIAS = "bias"
    DISABLE_AND_SET_VALUE = "disable_and_set_value"

    SCALE_STRENGTH = "scale_strength"

    TRANSFORM_CHANNEL = "transform_channel"
    SET_CHANNEL_VALUE = "set_channel_value"
    DISABLE_AND_SET_VALUES = "disable_and_set_values"


COLOR_CHANNELS = frozenset({MaterialChannel.BASE_COLOR})

SCALAR_CHANNELS = frozenset(
    {
        MaterialChannel.METALLIC,
        MaterialChannel.ROUGHNESS,
        MaterialChannel.SPECULAR,
        MaterialChannel.OPACITY,
    }
)

TEXTURE_CHANNELS = frozenset(
    {
        MaterialChannel.BASE_COLOR,
        MaterialChannel.METALLIC,
        MaterialChannel.ROUGHNESS,
        MaterialChannel.OPACITY,
        MaterialChannel.NORMAL,
    }
)

ALL_MATERIAL_CHANNELS = frozenset(
    {
        MaterialChannel.BASE_COLOR,
        MaterialChannel.METALLIC,
        MaterialChannel.ROUGHNESS,
        MaterialChannel.SPECULAR,
        MaterialChannel.OPACITY,
        MaterialChannel.NORMAL,
        MaterialChannel.ORM,
    }
)

CONSTANT_REQUIRED_OPS = frozenset(
    {
        RandomizationOp.SET_VALUE,
        RandomizationOp.CHOICE,
        RandomizationOp.SAMPLE_UNIFORM,
        RandomizationOp.JITTER_RELATIVE,
        RandomizationOp.CLAMP,
    }
)

CONSTANT_PLANNED_OPS = frozenset()

VALUE_TEXTURE_REQUIRED_OPS = frozenset(
    {
        RandomizationOp.SKIP,
        RandomizationOp.REPLACE,
        RandomizationOp.SCALE,
        RandomizationOp.BIAS,
        RandomizationOp.DISABLE_AND_SET_VALUE,
    }
)
VALUE_TEXTURE_PLANNED_OPS = frozenset()

PACKED_SCALAR_CHANNEL_REQUIRED_OPS = frozenset(
    {
        RandomizationOp.SKIP,
        RandomizationOp.TRANSFORM_CHANNEL,
        RandomizationOp.SET_CHANNEL_VALUE,
    }
)
PACKED_ORM_REQUIRED_OPS = frozenset(
    {
        RandomizationOp.SKIP,
        RandomizationOp.REPLACE,
        RandomizationOp.DISABLE_AND_SET_VALUES,
    }
)


def _build_implemented_operation_matrix() -> dict[tuple[str, str], frozenset[str]]:
    matrix: dict[tuple[str, str], frozenset[str]] = {}
    matrix[(MaterialChannel.BASE_COLOR, ChannelSource.CONSTANT)] = CONSTANT_REQUIRED_OPS | frozenset(
        {RandomizationOp.HSV_JITTER}
    )
    for channel in (
        MaterialChannel.ROUGHNESS,
        MaterialChannel.METALLIC,
        MaterialChannel.SPECULAR,
        MaterialChannel.OPACITY,
    ):
        matrix[(channel, ChannelSource.CONSTANT)] = CONSTANT_REQUIRED_OPS
    for channel in (
        MaterialChannel.BASE_COLOR,
        MaterialChannel.ROUGHNESS,
        MaterialChannel.METALLIC,
        MaterialChannel.OPACITY,
    ):
        matrix[(channel, ChannelSource.TEXTURE)] = VALUE_TEXTURE_REQUIRED_OPS
    matrix[(MaterialChannel.NORMAL, ChannelSource.TEXTURE)] = frozenset(
        {
            RandomizationOp.SKIP,
            RandomizationOp.REPLACE,
            RandomizationOp.SCALE_STRENGTH,
        }
    )
    for channel in (MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC):
        matrix[(channel, ChannelSource.PACKED_TEXTURE)] = PACKED_SCALAR_CHANNEL_REQUIRED_OPS
    matrix[(MaterialChannel.ORM, ChannelSource.PACKED_TEXTURE)] = PACKED_ORM_REQUIRED_OPS
    return matrix


def _build_planned_operation_matrix() -> dict[tuple[str, str], frozenset[str]]:
    matrix: dict[tuple[str, str], frozenset[str]] = {}
    matrix[(MaterialChannel.BASE_COLOR, ChannelSource.CONSTANT)] = (
        CONSTANT_REQUIRED_OPS | CONSTANT_PLANNED_OPS | frozenset({RandomizationOp.HSV_JITTER})
    )
    for channel in (MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC):
        matrix[(channel, ChannelSource.CONSTANT)] = CONSTANT_REQUIRED_OPS | CONSTANT_PLANNED_OPS
    for channel in (MaterialChannel.SPECULAR, MaterialChannel.OPACITY):
        matrix[(channel, ChannelSource.CONSTANT)] = CONSTANT_REQUIRED_OPS | CONSTANT_PLANNED_OPS
    matrix[(MaterialChannel.BASE_COLOR, ChannelSource.TEXTURE)] = frozenset(
        {
            RandomizationOp.SKIP,
            RandomizationOp.SCALE,
            RandomizationOp.REPLACE,
            RandomizationOp.BIAS,
            RandomizationOp.DISABLE_AND_SET_VALUE,
        }
    )
    for channel in (
        MaterialChannel.ROUGHNESS,
        MaterialChannel.METALLIC,
        MaterialChannel.OPACITY,
    ):
        matrix[(channel, ChannelSource.TEXTURE)] = VALUE_TEXTURE_REQUIRED_OPS | VALUE_TEXTURE_PLANNED_OPS
    matrix[(MaterialChannel.NORMAL, ChannelSource.TEXTURE)] = frozenset(
        {
            RandomizationOp.SKIP,
            RandomizationOp.REPLACE,
            RandomizationOp.SCALE_STRENGTH,
        }
    )
    for channel in (MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC):
        matrix[(channel, ChannelSource.PACKED_TEXTURE)] = PACKED_SCALAR_CHANNEL_REQUIRED_OPS
    matrix[(MaterialChannel.ORM, ChannelSource.PACKED_TEXTURE)] = PACKED_ORM_REQUIRED_OPS
    return matrix


IMPLEMENTED_OPERATION_MATRIX = _build_implemented_operation_matrix()
PLANNED_OPERATION_MATRIX = _build_planned_operation_matrix()


def implemented_ops(channel: str, source_type: str) -> frozenset[str]:
    return IMPLEMENTED_OPERATION_MATRIX.get((channel, source_type), frozenset())


def planned_ops(channel: str, source_type: str) -> frozenset[str]:
    return PLANNED_OPERATION_MATRIX.get((channel, source_type), frozenset())


def is_slotwise_safe_operation(source_type: str, operation: str) -> bool:
    if operation == RandomizationOp.SKIP:
        return True
    if source_type == ChannelSource.CONSTANT:
        return operation in {RandomizationOp.JITTER_RELATIVE, RandomizationOp.CLAMP}
    if source_type == ChannelSource.TEXTURE:
        return operation in {
            RandomizationOp.SCALE,
            RandomizationOp.BIAS,
            RandomizationOp.SCALE_STRENGTH,
        }
    if source_type == ChannelSource.PACKED_TEXTURE:
        return operation == RandomizationOp.TRANSFORM_CHANNEL
    return False
