import cutlass.cute as cute
import torch
from typing import no_type_check
import os

torch.manual_seed(42)

os.environ['CUTE_DSL_LOG_TO_CONSOLE'] = '1'
os.environ['CUTE_DSL_LOG_LEVEL'] = '10'

class TiledMatmul:

    def __init__(self, tile_width=12):
        self.tile_width = tile_width

    @no_type_check
    @cute.kernel
    def _kernel(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):

        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        row = bidy * self.tile_width + tidy
        col = bidx * self.tile_width + tidx

        smem = cute.arch.get_dyn_smem(cute.Float32)
        tile_layout = cute.make_layout((self.tile_width, self.tile_width))

        tileA = cute.make_tensor(smem, tile_layout)
        tileB = cute.make_tensor(smem + self.tile_width * self.tile_width, tile_layout)

        M, K = A.shape
        _, N = B.shape

        acc = cute.Float32(0.0)
        num_phases = (K + self.tile_width - 1) // self.tile_width

        for p in range(num_phases, uroll_full=True):
            if row < M and p * self.tile_width + tidx < K:
                tileA[tidy, tidx] = A[row, p * self.tile_width + tidx]
            else:
                tileA[tidy,tidx] = cute.Float32(0.0)

            if p * self.tile_width + tidy < K and col < N:
                tileB[tidy, tidx] = B[p * self.tile_width + tidy, col]
            else:
                tileB[tidy, tidx] = cute.Float32(0.0)

            cute.arch.sync_threads()

            for k in range(self.tile_width):
                acc += tileA[tidy, k] * tileB[k, tidx]
            
            cute.arch.sync_threads()

        if row < M and col < N:
            C[row, col] = cute.Float32(acc)



    @no_type_check
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):

        M, N = mC.shape
        threads_per_block = (self.tile_width, self.tile_width, 1)
        num_blocks = (
            (N + self.tile_width - 1) // self.tile_width,
            (M + self.tile_width - 1) // self.tile_width,
            1
        )

        smem_bytes = 2 * self.tile_width * self.tile_width * 4

        self._kernel(mA, mB, mC).launch(
            grid=num_blocks,
            block=threads_per_block,
            smem=smem_bytes
        )
        





        




