"""Track the crop offset of each accepted API chunk across async refills."""
from dataclasses import dataclass
from manaenv.policy.chunk_scheduler.base import REFILL_CONTEXT_KEY
from manaenv.policy.wire_interceptors import WallXRtc, WallXRtcCFG


class WallXApiRtc(WallXRtc):
    def __init__(self):
        super().__init__()
        self._source_offsets = {}
        self._request_delays = {}

    def on_request(self, request, *, env_idx):
        key = self._env_key(env_idx)
        request = dict(request)
        context = request.get(REFILL_CONTEXT_KEY)
        if isinstance(context, dict):
            context = dict(context)
            context['advanced_rows'] = context.get('advanced_rows', 0) + self._source_offsets.get(key, 0)
            request[REFILL_CONTEXT_KEY] = context
        result = super().on_request(request, env_idx=env_idx)
        obs = result.get('obs', {})
        if obs.get('use_rtc'):
            obs['s_model'] = int(context['advanced_rows'])
            # Preserve the command anchor, not the measured (possibly lagged) pose.
            with self._state_lock:
                obs['prev_action_chunk'][0] = self._previous_chunks[key][obs['s_model']]
        # AsyncChunkScheduler crops these committed rows before execution.
        self._request_delays[key] = int(obs.get('inference_delay', 0)) if obs.get('use_rtc') else 0
        return result

    def on_response(self, raw, *, env_idx):
        key = self._env_key(env_idx)
        with self._state_lock:
            current = self._pending_generations.get(key) == self._env_generations.get(key, 0)
        result = super().on_response(raw, env_idx=env_idx)
        if current:
            self._source_offsets[key] = self._request_delays.pop(key, 0)
        return result

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._source_offsets.clear()
            self._request_delays.clear()
        else:
            for i in env_ids:
                self._source_offsets.pop(int(i), None)
                self._request_delays.pop(int(i), None)


@dataclass
class WallXApiRtcCFG(WallXRtcCFG):
    max_delay: int = 4
    class_type: type = WallXApiRtc
