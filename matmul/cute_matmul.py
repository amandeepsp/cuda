import cutlass.cute as cute
from typing import no_type_check


class NaiveMatmul:
    @no_type_check
    @cute.kernel
    def _kernel(self, gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        bdimx, bdimy, _ = cute.arch.block_dim()

        row = bidy * bdimy + tidy
        col = bidx * bdimx + tidx

        M, K = gA.shape
        _, N = gB.shape

        if row < M and col < N:
            acc = cute.Float32(0.0)
            for k in range(K):
                acc += gA[row, k] * gB[k, col]
            gC[row, col] = cute.Float32(acc)

    @no_type_check
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        assert mA.shape[1] == mB.shape[0]

        M, N = mC.shape

        threads_per_block = (16, 16, 1)
        num_blocks = (
            (N + threads_per_block[0] - 1) // threads_per_block[0],
            (M + threads_per_block[1] - 1) // threads_per_block[1],
            1,
        )

        self._kernel(mA, mB, mC).launch(
            grid=num_blocks,
            block=threads_per_block,
        )
