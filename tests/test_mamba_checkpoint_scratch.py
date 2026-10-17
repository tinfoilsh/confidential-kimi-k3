"""Check that prefix-cache hits refer to written recurrent states."""

from types import SimpleNamespace
import unittest

import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

MAMBA_BLOCK_SIZE = 1664
HASH_BLOCK_SIZE = 64
CHECKPOINT_ALIGNMENT = 16
MAMBA_GROUP = 1
PROMPT_TOKENS = 26978
MAX_CONTEXT = 262144
BLOCK_COUNT = 10000
DRAFT_WINDOW = 2048
TARGET_HEAD_SIZE = 576
DRAFT_HEAD_SIZE = 64
DRAFT_KV_HEADS = 2
SPECULATIVE_DEPTHS = (0, 1, 8)
CHUNK_BUDGETS = (4096, 8192, 16384)
RETENTION_INTERVALS = (None, 0)


class MambaCheckpointScratchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_none_hash(sha256)

    def make_request(self, request_id):
        return Request(
            request_id=request_id,
            prompt_token_ids=[1] * PROMPT_TOKENS,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(HASH_BLOCK_SIZE, sha256),
        )

    def check_prefill(self, depth, budget, retention, drop_eagle_block):
        recurrent = MambaSpec(
            block_size=MAMBA_BLOCK_SIZE,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=depth,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=CHECKPOINT_ALIGNMENT,
        )
        config = KVCacheConfig(
            num_blocks=BLOCK_COUNT,
            kv_cache_tensors=[],
            prefix_cache_retention_interval=retention,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["target.attention"],
                    MLAAttentionSpec(
                        block_size=MAMBA_BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=TARGET_HEAD_SIZE,
                        dtype=torch.float8_e4m3fn,
                        cache_dtype_str="fp8",
                    ),
                ),
                KVCacheGroupSpec(["target.recurrent"], recurrent),
                KVCacheGroupSpec(
                    ["draft.attention"],
                    SlidingWindowSpec(
                        block_size=HASH_BLOCK_SIZE,
                        num_kv_heads=DRAFT_KV_HEADS,
                        head_size=DRAFT_HEAD_SIZE,
                        dtype=torch.float8_e4m3fn,
                        sliding_window=DRAFT_WINDOW,
                    ),
                ),
            ],
        )
        manager = KVCacheManager(
            config,
            max_model_len=MAX_CONTEXT,
            scheduler_block_size=MAMBA_BLOCK_SIZE,
            hash_block_size=HASH_BLOCK_SIZE,
            enable_caching=True,
            use_eagle=drop_eagle_block,
        )
        scheduler = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=HASH_BLOCK_SIZE),
            mamba_state_block_size=MAMBA_BLOCK_SIZE,
            use_eagle_block_drop=drop_eagle_block,
            hash_block_size=HASH_BLOCK_SIZE,
            max_num_scheduled_tokens=budget,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            mamba_has_prefill_checkpoint_blocks=True,
            mamba_prefill_checkpoint_alignment=CHECKPOINT_ALIGNMENT,
            mamba_partial_cache_hit=True,
            mamba_shared_prefix_checkpoint=False,
        )
        request = self.make_request("cache-producer")
        mamba = manager.coordinator.single_type_managers[MAMBA_GROUP]
        written_states = {}
        checked = 0
        while request.num_computed_tokens < PROMPT_TOKENS:
            start = request.num_computed_tokens
            count = Scheduler._mamba_block_aligned_split(
                scheduler, request, min(budget, PROMPT_TOKENS - start)
            )
            self.assertGreater(count, 0)
            end = start + count
            self.assertIsNotNone(
                manager.allocate_slots(request, count, num_lookahead_tokens=depth)
            )
            blocks = mamba.req_to_blocks[request.request_id]
            running_column = (end - 1) // MAMBA_BLOCK_SIZE
            written_states[blocks[running_column].block_id] = end
            offsets, columns = compute_mamba_prefill_checkpoints(
                [end], [count], HASH_BLOCK_SIZE, MAMBA_BLOCK_SIZE,
                CHECKPOINT_ALIGNMENT, drop_eagle_block,
            )
            if offsets[0] and not blocks[columns[0]].is_null:
                written_states[blocks[columns[0]].block_id] = start + offsets[0]
            request.num_computed_tokens = end

            # Model the worker's state writes, then check the real allocator's
            # published hashes against the token position those writes cover.
            for block in blocks:
                if block.is_null or block.block_hash is None:
                    continue
                self.assertEqual(
                    written_states.get(block.block_id),
                    block.block_hash_num_tokens,
                    f"Unwritten or stale state published at {end} tokens",
                )
                checked += 1
        self.assertGreater(checked, 0)
        manager.free(request)
        manager.new_step_starts()
        cached, num_cached, _ = manager.get_computed_blocks(
            self.make_request("cache-consumer")
        )
        if not drop_eagle_block or retention is None:
            self.assertGreater(num_cached, 0)
        if num_cached:
            self.assertEqual(
                written_states[cached.blocks[MAMBA_GROUP][-1].block_id], num_cached
            )

    def test_checkpoint_does_not_publish_speculative_scratch(self):
        for depth in SPECULATIVE_DEPTHS:
            for budget in CHUNK_BUDGETS:
                for retention in RETENTION_INTERVALS:
                    with self.subTest(depth=depth, budget=budget, retention=retention):
                        self.check_prefill(depth, budget, retention, True)

    def test_independent_drafter_reuses_sparse_checkpoint(self):
        for budget in CHUNK_BUDGETS:
            with self.subTest(budget=budget):
                self.check_prefill(max(SPECULATIVE_DEPTHS), budget, 0, False)


if __name__ == "__main__":
    unittest.main()
