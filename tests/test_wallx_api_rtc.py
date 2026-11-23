import unittest
import numpy as np
from manaenv.policy.wallx_api_rtc import WallXApiRtc, WallXApiRtcCFG
from manaenv.policy.chunk_scheduler.base import REFILL_CONTEXT_KEY


class Tests(unittest.TestCase):
    def test_multiple_async_crops_keep_source_offset(self):
        hook = WallXApiRtc(); hook.load(WallXApiRtcCFG())
        state = dict(follow1_pos=np.zeros(7), follow2_pos=np.zeros(7))
        def request():
            return {'obs':{'state':state}, REFILL_CONTEXT_KEY:dict(crop_overlap=True, advanced_rows=2, overlap_rows=4)}
        hook.on_request({'obs':{'state':state}}, env_idx=0)
        rows = np.repeat(np.arange(33, dtype=np.float32)[:,None],14,axis=1)
        hook.on_response({'actions':rows}, env_idx=0)
        first = hook.on_request(request(), env_idx=0)['obs']
        self.assertEqual(first['inference_delay'],4)
        self.assertEqual(first['s_model'],2)
        np.testing.assert_equal(first['prev_action_chunk'][0],rows[2])
        np.testing.assert_equal(first['prev_action_chunk'][1],rows[3])
        rows2 = rows+100
        hook.on_response({'actions':rows2}, env_idx=0)
        second = hook.on_request(request(), env_idx=0)['obs']
        # Four rows were cropped, then two executed: next committed row is 7.
        self.assertEqual(second['s_model'],6)
        np.testing.assert_equal(second['prev_action_chunk'][1],rows2[7])
        hook.reset()
        self.assertNotIn('use_rtc',hook.on_request(request(),env_idx=0)['obs'])

