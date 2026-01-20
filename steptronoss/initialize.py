# Copyright (c) 2025, STEPFUN INC. All rights reserved.

import random

import numpy as np
import torch
import torch.distributed
from loguru import logger

from steptronoss.core.parallel_state import PM


def mtp_initialize(models):
    torch.distributed.barrier()
    from steptronoss.core.parallel_state import PM
    from steptronoss.utils.utils import unwrap_model

    model_may_need_sync = [model for model in models if unwrap_model(model).need_mtp_sync]
    assert len(model_may_need_sync) <= 1, "we currently do not support multiple embeddings on the same rank"
    # build new group for mtp
    if len(model_may_need_sync) == 1:
        tensor_in = torch.tensor(
            torch.distributed.get_rank(),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
    else:
        tensor_in = torch.tensor(-1, dtype=torch.int64, device=torch.cuda.current_device())
    tensor_out = -1 * torch.ones(PM.size_of("PP"), dtype=torch.int64, device=torch.cuda.current_device())
    # gather in pp group, and the ranks with lm_embeddings will send its rank otherwise send -1
    torch.distributed.all_gather_into_tensor(tensor_out, tensor_in, group=PM.group_of("PP"))
    emb_sync_indices = (tensor_out != -1).nonzero(as_tuple=True)[0].tolist()
    emb_sync_residual_indices = (tensor_out == -1).nonzero(as_tuple=True)[0].tolist()
    emb_sync_ranks = []
    emb_sync_residual_ranks = []
    # in each pp group, the ranks with lm_embeddings are on the same stages, therefore we can get the ranks with lm_embeddings in the world
    for pp_ranks in PM.world_ranks_of("PP"):
        emb_sync_ranks.append([pp_ranks[i] for i in emb_sync_indices])
        emb_sync_residual_ranks.append([pp_ranks[i] for i in emb_sync_residual_indices])

    # since PM.new_parallel() does not support ranks not in any groups, we creaste a residul group
    PM.new_parallel("EMB_SYNC", ranks=emb_sync_ranks + emb_sync_residual_ranks)

    if len(model_may_need_sync) == 1:
        unwrap_model(model_may_need_sync[0]).sync_tok_emb_param()
    torch.distributed.barrier(group=PM.group_of("EMB_SYNC"))
    logger.info("Successfully sync input embeddings.")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_mpu_random_seed(seed_):
    logger.info(f"> setting random seeds to {seed_} ...", at=0)

    from steptronoss.core.tensor_parallel.random import (
        _CUDA_RNG_STATE_TRACKER,
        _EXPERT_MODEL_PARALLEL_RNG_TRACKER_NAME,
        _MODEL_PARALLEL_RNG_TRACKER_NAME,
    )

    seed = seed_ + (100 * PM.rank_in("PP"))
    # Ensure different data parallel ranks get different seeds
    set_random_seed(seed)
    if torch.cuda.device_count() == 0:
        return

    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tp_rank_specific_seed = offset + PM.rank_in("TP")

    # For expert layers, create a unique seed for each (ep, etp) pair.
    # Use a different offset than attention tp.
    expert_offset = seed + 3718
    etp_rank_specific_seed = expert_offset + PM.rank_in("ETP") + PM.rank_in("EP") * PM.size_of("ETP")
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tp_rank_specific_seed)
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_MODEL_PARALLEL_RNG_TRACKER_NAME, etp_rank_specific_seed)
