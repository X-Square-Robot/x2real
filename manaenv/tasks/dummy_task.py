"""Mana Dummy Task - minimal task for testing without specific objectives."""

# IsaacLab Arena 3.0 deleted ``tasks.dummy_task.DummyTask``; the null-object task
# is now ``tasks.no_task.NoTask``. ManaDummyTask only adds the
# manaenv-required get_prompt().
from isaaclab_arena.tasks.no_task import NoTask


class ManaDummyTask(NoTask):
    def get_prompt(self) -> str:
        if hasattr(self, "_prompt") and self._prompt:
            return self._prompt
        return "Perform the task."
