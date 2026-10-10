"""Policy namespace compatibility for chunk connector helpers."""

from manaenv.chunk.connectors import blend_chunk_start_to_last_command


def connect_action_chunk(chunk, *, last_command, slot_layout, ee_frames=None, device=None):
    """Blend the next action chunk boundary using the shared chunk helper."""

    del ee_frames, device
    return blend_chunk_start_to_last_command(
        chunk,
        last_command=last_command,
        slot_layout=slot_layout,
    )
