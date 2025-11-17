import torch
from cutlass.cute.runtime import from_dlpack

from tiled_matmul import TiledMatmul
from naive_matmul import NaiveMatmul
from fmma_tiled_matmul import FusedTiledMatmul

torch.manual_seed(42)

M, L, N = 2000, 1000, 2000

a = torch.randn(M, L, device="cuda", dtype=torch.float32)
b = torch.randn(L, N, device="cuda", dtype=torch.float32)
c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
bT_ = from_dlpack(b.T, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

naive_matmul = NaiveMatmul()
tiled_matmul = TiledMatmul(tile_width=16)
cute_tiled_matmul = FusedTiledMatmul()

naive_matmul(a_, b_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5)

tiled_matmul(a_, b_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5)

cute_tiled_matmul(a_, bT_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5)

