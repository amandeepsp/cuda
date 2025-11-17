import cutlass.cute as cute
from cutlass.utils.smem_allocator import SmemAllocator
import cutlass
from typing import Type, no_type_check
import os

os.environ["CUTE_DSL_PRINT_PTX"] = "1"

# TODOS:
# 1. Tune block_tiler, copy layouts, and mma atom layout for better performance
# 2. Support different data types
# 3. Add predication for non square matrices
# 4. cpasync support for better performance
# 5. Pipelining of copy and mma ops
# 6. Support NN layout for inputs


class FusedTiledMatmul:
    def __init__(self, dtype: Type[cute.Numeric] = cute.Float32):
        self.dtype = dtype

    @no_type_check
    @cute.kernel
    def _kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        block_tiler: cutlass.Shape,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        cta_coords = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)

        gA = cute.local_tile(mA, block_tiler, cta_coords, proj=(1, None, 1))
        gB = cute.local_tile(mB, block_tiler, cta_coords, proj=(None, 1, 1))
        gC = cute.local_tile(mC, block_tiler, cta_coords, proj=(1, 1, None))

        smem = SmemAllocator()

        sA = smem.allocate_tensor(cute.Float32, sA_layout)
        sB = smem.allocate_tensor(cute.Float32, sB_layout)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        # Clear the accumulator
        tCrC.fill(0.0)

        k_tiles = cute.size(tAgA, mode=[3])

        for k in range(k_tiles):
            # Copy rmem to smem for the current k-tile
            cute.copy(tiled_copy_A, tAgA[None, None, None, k], tAsA)
            cute.copy(tiled_copy_B, tBgB[None, None, None, k], tBsB)
            cute.arch.sync_threads()

            cute.gemm(tiled_mma, tCrC, tCsA, tCsB, tCrC)
            cute.arch.sync_threads()

        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC)

    @no_type_check
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        block_tiler = (128, 128, 8)  # (block_m, block_n, block_k)

        sA_layout = cute.make_layout(
            (block_tiler[0], block_tiler[2])
        )  # (block_m, block_k)
        sB_layout = cute.make_layout(
            (block_tiler[1], block_tiler[2])
        )  # (block_n, block_k)

        copy_atom_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cute.Float32)
        copy_atom_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cute.Float32)

        tiled_copy_A = cute.make_tiled_copy_tv(
            copy_atom_A,
            thr_layout=cute.make_layout((32, 8)),
            val_layout=cute.make_layout((4, 1)),
        )

        tiled_copy_B = cute.make_tiled_copy_tv(
            copy_atom_B,
            thr_layout=cute.make_layout((32, 8)),
            val_layout=cute.make_layout((4, 1)),
        )

        atoms_layout_mnk = cute.make_layout((16, 16, 1))

        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.MmaUniversalOp(abacc_dtype=cute.Float32),
            atom_layout_mnk=atoms_layout_mnk,
        )

        grid_dim = *cute.ceil_div(mC.shape, (block_tiler[0], block_tiler[1])), 1

        self._kernel(
            mA,
            mB,
            mC,
            block_tiler,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=grid_dim,
            block=(cute.size(atoms_layout_mnk), 1, 1),
        )
