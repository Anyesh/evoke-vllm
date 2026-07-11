"""RoPE-delta rotation for landing a recovered K block at a position other
than the one it was written at.

Dormant in P1 (design spec 01a section 3). The stock vLLM offload path only
restores a block to the logical position it was hashed at, so
``original_position`` always equals the new position and no rotation is ever
needed: ``EvokeOffloadingManager`` and ``EvokeCachePolicy`` never call into
this module. It exists so that wiring the RFC-track smart-recovery selector
(design spec 03), which can land a block at a different position, is a
configuration change rather than a rewrite. Shipping it as dead, unimported
code would be dishonest; shipping it as an inert, documented module is not.

No rotation math is implemented in this scaffold.
"""

from __future__ import annotations

import torch


class EvokeRopeDeltaRotator:
    def __init__(
        self,
        k_views_per_layer: list[torch.Tensor],
        cos_sin_cache: torch.Tensor,
        head_size: int,
        is_neox: bool = True,
    ) -> None:
        raise NotImplementedError

    def rotate(
        self,
        block_indices: list[int],
        original_positions: list[int],
        new_positions: list[int],
    ) -> None:
        raise NotImplementedError
