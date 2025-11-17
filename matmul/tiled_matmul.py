import cutlass.cute as cute
from typing import no_type_check, Type
import cutlass


class TiledMatmul:
    def __init__(
        self,
        tile_width=16,
        dtype: Type[cutlass.Numeric] = cutlass.Float32,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.tile_width = tile_width
        self.dtype = dtype
        self.acc_dtype = acc_dtype

    @no_type_check
    @cute.kernel
    def _kernel(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        row = bidy * self.tile_width + tidy
        col = bidx * self.tile_width + tidx

        smem = cutlass.utils.SmemAllocator()
        tiles = smem.allocate(self.shared_storage)
        tile_layout = cute.make_layout((self.tile_width, self.tile_width))

        tileA = tiles.tileA.get_tensor(tile_layout)
        tileB = tiles.tileB.get_tensor(tile_layout)

        M, K = A.shape
        _, N = B.shape

        acc = self.acc_dtype(0.0)
        num_phases = (K + self.tile_width - 1) // self.tile_width

        for p in range(num_phases):
            if row < M and p * self.tile_width + tidx < K:
                tileA[tidy, tidx] = A[row, p * self.tile_width + tidx]
            else:
                tileA[tidy, tidx] = self.dtype(0.0)

            if p * self.tile_width + tidy < K and col < N:
                tileB[tidx, tidy] = B[p * self.tile_width + tidy, col]
            else:
                tileB[tidx, tidy] = self.dtype(0.0)

            cute.arch.sync_threads()

            for k in range(self.tile_width):
                acc += tileA[tidy, k] * tileB[tidx, k]

            cute.arch.sync_threads()

        if row < M and col < N:
            C[row, col] = self.dtype(acc)

    @no_type_check
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        tile_size = self.tile_width * self.tile_width

        @cute.struct
        class SharedStorage:
            tileA: cute.struct.MemRange[self.dtype, tile_size] # type: ignore
            tileB: cute.struct.MemRange[self.dtype, tile_size] # type: ignore

        self.shared_storage = SharedStorage

        M, N = mC.shape
        threads_per_block = (self.tile_width, self.tile_width, 1)
        num_blocks = (
            (N + self.tile_width - 1) // self.tile_width,
            (M + self.tile_width - 1) // self.tile_width,
            1,
        )

        self._kernel(mA, mB, mC).launch(
            grid=num_blocks, block=threads_per_block, smem=SharedStorage.size_in_bytes()
        )
