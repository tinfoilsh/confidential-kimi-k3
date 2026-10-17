"""Check real scheduler chunk boundaries for mixed draft and recurrent caches."""

from types import SimpleNamespace
import unittest

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import Request

DRAFT_BLOCK_SIZE = 64
MAMBA_BLOCK_SIZE = 1664
PREFILL_BUDGET = 8192
PROMPT_TOKENS = 26978
OUTPUT_LIMIT = 1024


class MambaChunkAlignmentTest(unittest.TestCase):
    def split(self, cache_block_size, start):
        spec = MambaSpec(
            block_size=MAMBA_BLOCK_SIZE,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
        scheduler = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=cache_block_size),
            mamba_state_block_size=spec.block_size,
            use_eagle_block_drop=True,
            hash_block_size=DRAFT_BLOCK_SIZE,
            max_num_scheduled_tokens=PREFILL_BUDGET,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            mamba_has_prefill_checkpoint_blocks=False,
            mamba_prefill_checkpoint_alignment=None,
            mamba_partial_cache_hit=True,
            mamba_shared_prefix_checkpoint=False,
        )
        request = Request(
            request_id="prefix-resume",
            prompt_token_ids=[1] * PROMPT_TOKENS,
            sampling_params=SamplingParams(max_tokens=OUTPUT_LIMIT),
            pooling_params=None,
        )
        request.num_computed_tokens = start
        return Scheduler._mamba_block_aligned_split(
            scheduler, request, PREFILL_BUDGET
        )

    def test_prefill_ends_on_a_recurrent_state_boundary(self):
        expected = PREFILL_BUDGET // MAMBA_BLOCK_SIZE * MAMBA_BLOCK_SIZE
        for cache_block_size in (DRAFT_BLOCK_SIZE, MAMBA_BLOCK_SIZE):
            with self.subTest(cache_block_size=cache_block_size):
                self.assertEqual(self.split(cache_block_size, 0), expected)

    def test_partial_resume_stops_at_the_next_recurrent_state_boundary(self):
        start = MAMBA_BLOCK_SIZE + DRAFT_BLOCK_SIZE
        expected = MAMBA_BLOCK_SIZE - DRAFT_BLOCK_SIZE
        for cache_block_size in (DRAFT_BLOCK_SIZE, MAMBA_BLOCK_SIZE):
            with self.subTest(cache_block_size=cache_block_size):
                self.assertEqual(self.split(cache_block_size, start), expected)


if __name__ == "__main__":
    unittest.main()
