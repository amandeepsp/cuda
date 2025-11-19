from task import input_t, output_t
from typing import Type, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.typing import Pointer
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr


class Sm100BlockScaledGemv:
    def __init__(
        self,
        mma_tiler_mnk: Tuple[int, int, int],
        ab_dtype: Type[cute.Numeric] = cute.Float4E2M1FN,
        sf_dtype: Type[cute.Numeric] = cute.Float8E4M3FN,
        c_dtype: Type[cute.Numeric] = cute.Float16,
    ):
        self.mma_tiler_mnk = mma_tiler_mnk
        self.ab_dtype = ab_dtype
        self.sf_dtype = sf_dtype
        self.c_dtype = c_dtype
        self.sf_vec_size = 16
        self.n_padded = 128

    @cute.jit
    def __call__(
        self,
        a_ptr: Pointer,
        b_ptr: Pointer,
        sfa_ptr: Pointer,
        sfb_ptr: Pointer,
        c_ptr: Pointer,
        problem_size: Tuple[int, int, int, int],
    ):
        m, _, k, l = problem_size

        a_tensor = cute.make_tensor(
            a_ptr, layout=cute.make_layout((m, k, l), stride=(k, 1, m * k))
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_layout(
                (self.n_padded, k, l), stride=(k, 1, self.n_padded * k)
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout((m, 1, l), stride=((1, l, m)))
        )

        # Convert scale factor tensors to MMA layout
        # The layout matches Tensor Core requirements: (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        raise NotImplementedError

    @cute.kernel
    def _kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        mC: cute.Tensor,
    ):
        pass


# Global cache for compiled kernel
_compiled_kernel_cache = None


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMV kernel.

    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.

    Args:
        data: Tuple of (a, b, sfa_cpu, sfb_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn
            b: [1, k, l] - Input vector in float4e2m1fn
            sfa_cpu: [m, k, l] - Scale factors in float8_e4m3fn
            sfb_cpu: [1, k, l] - Scale factors in float8_e4m3fn
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            c: [m, 1, l] - Output vector in float16

    Returns:
        Output tensor c with computed GEMV results
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data
    stream = cutlass_torch.default_stream()

    # Ensure kernel is compiled (will use cached version if available)
    # To avoid the compilation overhead, we compile the kernel once and cache it.
    global _compiled_kernel_cache

    gemm = Sm100BlockScaledGemv(mma_tiler_mnk=(128, 1, 64))

    if _compiled_kernel_cache is None:
        # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
        a_ptr = make_ptr(gemm.ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        c_ptr = make_ptr(gemm.c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        sfa_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
        sfb_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

        # Configure gemm kernel

        # Compile the kernel
        _compiled_kernel_cache = cute.compile(
            gemm, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0), 1, stream
        )

    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2
    # GEMV N dimension is always 1
    n = 128

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        gemm.ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        gemm.ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        gemm.c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        gemm.sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        gemm.sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    _compiled_kernel_cache(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l), stream)

    return c
