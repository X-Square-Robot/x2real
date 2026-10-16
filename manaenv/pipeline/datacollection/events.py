"""Semantic datacollection events and dispatch helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

RESET_EPISODE = "reset_episode"
MARK_SUCCESS = "mark_success"
START_RECORDING = "start_recording"
STOP_RECORDING = "stop_recording"
TOGGLE_LAYOUT_EDIT = "toggle_layout_edit"

DEFAULT_COLLECTION_EVENT_MAP: dict[str, str] = {
    "R": RESET_EPISODE,
    "ENTER": MARK_SUCCESS,
    "START": START_RECORDING,
    "STOP": STOP_RECORDING,
    "RESET": RESET_EPISODE,
    "L": TOGGLE_LAYOUT_EDIT,
}


@dataclass(frozen=True)
class CollectionEvent:
    """A semantic event emitted by a planner or input device."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionInputFrame:
    """One sampled input frame: action plus semantic datacollection events."""

    action: Any
    events: list[CollectionEvent] = field(default_factory=list)


class CollectionEventQueue:
    """Small FIFO queue shared by input adapters and collection loops."""

    def __init__(self) -> None:
        self._events: deque[CollectionEvent] = deque()

    def emit(self, name: str, **payload: Any) -> None:
        self._events.append(CollectionEvent(name=name, payload=dict(payload)))

    def drain(self) -> list[CollectionEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def bind_controller_callbacks(
        self,
        controller: Any,
        event_map: Mapping[str, str] | None = None,
    ) -> None:
        """Register controller callbacks that only emit semantic events."""

        add_callback = getattr(controller, "add_callback", None)
        if not callable(add_callback):
            return
        for key, event_name in (event_map or DEFAULT_COLLECTION_EVENT_MAP).items():
            add_callback(str(key), lambda name=event_name: self.emit(name))


EventHandler = Callable[[CollectionEvent], None]


class CollectionEventDispatcher:
    """Dispatch semantic events to loop-owned handlers."""

    def __init__(self, handlers: Mapping[str, EventHandler]) -> None:
        self._handlers = dict(handlers)

    def dispatch(self, events: Iterable[CollectionEvent]) -> None:
        for event in events:
            handler = self._handlers.get(event.name)
            if handler is not None:
                handler(event)


def drain_collection_events(source: Any) -> list[CollectionEvent]:
    drain = getattr(source, "drain_events", None)
    if not callable(drain):
        return []
    return list(drain())
