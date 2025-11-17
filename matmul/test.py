import torch
from cutlass.cute.runtime import from_dlpack

from tiled_matmul import TiledMatmul
from naive_matmul import NaiveMatmul
from fmma_tiled_matmul import FusedTiledMatmul

torch.manual_seed(42)

M, L, N = 8192, 4096, 8192

a = torch.randn(M, L, device="cuda", dtype=torch.float32)
b = torch.randn(L, N, device="cuda", dtype=torch.float32)
bT = b.t().contiguous()
c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

a_ = from_dlpack(a)
b_ = from_dlpack(b)
bT_ = from_dlpack(bT)
c_ = from_dlpack(c)

naive_matmul = NaiveMatmul()
tiled_matmul = TiledMatmul(tile_width=16)
cute_tiled_matmul = FusedTiledMatmul()

naive_matmul(a_, b_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5)

tiled_matmul(a_, b_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5)

cute_tiled_matmul(a_, bT_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5)

