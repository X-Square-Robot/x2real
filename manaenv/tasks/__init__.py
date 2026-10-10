"""Tasks module for manaenv.

All tasks are generated via the modular ``TaskGenerator`` pipeline.
Components declared in YAML ``terms`` sections produce IsaacLab
configclass instances. The runtime task type is
``isaaclab_arena.tasks.task_base.ManaTask`` (Arena-owned); the
``manaenv.tasks.mana_task`` compatibility wrapper has been removed in
favour of the unified Arena ``ManaTask`` API.
"""
