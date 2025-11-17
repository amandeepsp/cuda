import cutlass.cute as cute
from cutlass.utils.smem_allocator import SmemAllocator
import cutlass
from typing import Type, no_type_check, Tuple
import cutlass.utils as utils


class FusedTiledMatmul:
    def __init__(
        self,
        acc_dtype: Type[cute.Numeric] = cute.Float32,
        block_tiler: Tuple[int, int, int] = (128, 128, 8),
        num_threads=256,
        num_vectorize_copy=4,
    ):
        self.block_tiler = block_tiler
        self.num_threads = num_threads
        assert num_threads > 0 and num_threads % 16 == 0, (
            "num_threads must be positive and multiple of 16 for MMA thread layout"
        )

        self.b_m, self.b_n, self.b_k = block_tiler
        assert self.b_m % 16 == 0 and self.b_n % 16 == 0 and self.b_k % 4 == 0, (
            "block_tiler dimensions must be multiples of 16, 16, and 4 respectively"
        )

        self.acc_dtype = acc_dtype
        self.num_vectorize_copy = num_vectorize_copy

    @no_type_check
    @cute.kernel
    def _kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
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

        # Local tiles for current Thread Block
        # gA: (b_m, b_k, k_tiles), gB: (b_n, b_k, k_tiles), gC: (b_m, b_n)
        gA = cute.local_tile(mA, self.block_tiler, cta_coords, proj=(1, None, 1))
        gB = cute.local_tile(mB, self.block_tiler, cta_coords, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.block_tiler, cta_coords, proj=(1, 1, None))

        smem = SmemAllocator()

        # Allocate shared memory for A and B tiles
        # sA: (b_m, b_k), sB: (b_n, b_k)
        sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, byte_alignment=16)

        # Get tiles for current thread
        # tAgA: (cpy, cpy_m, cpy_k, k_tiles), tBgB: (cpy, cpy_n, cpy_k, k_tiles)
        # tAsA: (cpy, cpy_m, cpy_k), tBsB: (cpy, cpy_n, cpy_k)
        # cpy: copy index for vectorized copy - (atom_v, rest_v)
        # atom_v: index within vectorized copy
        # rest_v: number for remaining copies assigned to the thread
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        # Create predicate tensors for A and B
        # cA = identity tensor for sA - (b_m, b_k) -> (m_idx, k_idx)
        # cB = identity tensor for sB - (b_n, b_k) -> (n_idx, k_idx)
        # tAcA = current thread's partition of cA - (cpy, cpy_m, cpy_k) -> (m_idx, k_idx)
        # tAcB = current thread's partition of cB - (cpy, cpy_n, cpy_k) -> (n_idx, k_idx)
        # tApA = predicate tensor for all but last tAsA - (rest_v, cpy_m, cpy_k), stride=(cpy_m, 1, 0)
        # tBpB = predicate tensor for all but last tBsB - (rest_v, cpy_n, cpy_k), stride=(cpy_n, 1, 0)
        # tApA, tBpB are shared across all k_tiles since stride of k mode = 0, same values are used for all k
        # tApA_residue_k = predicate tensor for last tAsA - (rest_v, cpy_m, cpy_k), stride=(cpy_m * cpy_k, cpy_k, 1)
        # tBpB_residue_k = predicate tensor for last tBsB - (rest_v, cpy_n, cpy_k), stride=(cpy_n * cpy_k, cpy_k, 1)

        cA = cute.make_identity_tensor(sA_layout.shape)
        cB = cute.make_identity_tensor(sB_layout.shape)

        tcA = cute.local_tile(cA, self.block_tiler, cta_coords, proj=(1, None, 1))
        tcB = cute.local_tile(cB, self.block_tiler, cta_coords, proj=(None, 1, 1))

        tAcA = thr_copy_A.partition_S(tcA)
        tBcB = thr_copy_B.partition_S(tcB)
        # Allocate predicate tensors for m and n
        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        # Allocate predicate tensors for m, n and k for residue k-tile
        tApA_residue_k = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(
                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                    cute.size(tAsA, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        tBpB_residue_k = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(
                    cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                    cute.size(tBsB, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        # Set predicates for m/n bounds for mainloop
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )

        # Set predicates for m/n/k bounds for residue k tile
        for rest_v in range(tApA_residue_k.shape[0]):
            for m in range(tApA_residue_k.shape[1]):
                for k in range(tApA_residue_k.shape[2]):
                    coord_A = tAcA[(0, rest_v), m, k, 0]
                    tApA_residue_k[rest_v, m, k] = cute.elem_less(
                        (coord_A[0], coord_A[1]), (mA.shape[0], mA.shape[1])
                    )

        for rest_v in range(tBpB_residue_k.shape[0]):
            for n in range(tBpB_residue_k.shape[1]):
                for k in range(tBpB_residue_k.shape[2]):
                    coord_B = tBcB[(0, rest_v), n, k, 0]
                    tBpB_residue_k[rest_v, n, k] = cute.elem_less(
                        (coord_B[0], coord_B[1]), (mB.shape[0], mB.shape[1])
                    )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        # Clear the accumulator
        tCrC.fill(0.0)

        k_tiles = cute.size(tAgA, mode=[3])

        # Main k-tile loop
        # For each k-tile:
        #   Copy A and B tiles from GMEM to SMEM
        #   Sync threads
        #   Perform MMA on the current k-tile
        #   Sync threads

        for k in range(k_tiles):
            # Copy rmem to smem for the current k-tile

            # last k-tile, use residue k predication, might be partial tile in K dimension
            if k == k_tiles - 1:
                cute.copy(
                    tiled_copy_A, tAgA[None, None, None, k], tAsA, pred=tApA_residue_k
                )
                cute.copy(
                    tiled_copy_B, tBgB[None, None, None, k], tBsB, pred=tBpB_residue_k
                )
            else:
                cute.copy(tiled_copy_A, tAgA[None, None, None, k], tAsA, pred=tApA)
                cute.copy(tiled_copy_B, tBgB[None, None, None, k], tBsB, pred=tBpB)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            cute.gemm(tiled_mma, tCrC, tCsA, tCsB, tCrC)
            cute.arch.sync_threads()

        # Epilogue: copy accumulator to global memory with predication for m and n bounds
        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_fragment(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - self.b_m * bidx
        residue_n = mC.shape[1] - self.b_n * bidy
        for i in range(cute.size(tCrC.shape)):
            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=predC)

    @no_type_check
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
    ):
        # Shared memory layouts for A and B, both are m/n-major to vectorize copies from shared memory
        # to registers in MMA op.
        # sA = (b_m, b_k), sB = (b_n, b_k)
        sA_layout = cute.make_layout((self.b_m, self.b_k))
        sB_layout = cute.make_layout((self.b_n, self.b_k))

        # Copy Layouts for GMEM -> SMEM copy
        # number of elements to copy in a vectorized load/store

        tA = cute.make_layout(
            (self.num_threads // self.b_k, self.b_k), stride=(self.b_k, 1)
        )
        vA = cute.make_layout((1, 1))

        copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width,
        )

        num_vectorized = self.num_vectorize_copy if (mB.layout[0].max_alignment % 16 == 0) else 1
        copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mB.element_type.width * num_vectorized,
        )
        major_mode_size = self.b_n // num_vectorized
        tB = cute.make_layout(
            (major_mode_size, self.num_threads // major_mode_size),
            stride=(1, major_mode_size),
        )
        vB = cute.make_layout((num_vectorized, 1))


        tiled_copy_A = cute.make_tiled_copy_tv(
            copy_atom_A, thr_layout=tA, val_layout=vA
        )

        tiled_copy_B = cute.make_tiled_copy_tv(
            copy_atom_B, thr_layout=tB, val_layout=vB
        )

        # Layouts for MMA GEMM

        atoms_layout_mnk = cute.make_layout(
            (self.num_threads // 16, 16, 1), stride=(16, 1, 0)
        )

        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.MmaUniversalOp(abacc_dtype=self.acc_dtype),
            atom_layout_mnk=atoms_layout_mnk,
        )

        grid_dim = (
            *cute.ceil_div(mC.shape, (self.b_m, self.b_n)),
            1,
        )

        self._kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=grid_dim,
            block=(cute.size(atoms_layout_mnk), 1, 1),
        )
