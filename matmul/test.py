import torch
from cutlass.cute.runtime import from_dlpack

from cute_tiled_matmul import TiledMatmul
from cute_matmul import NaiveMatmul

torch.manual_seed(42)

print(torch.cuda.get_device_properties(0))

M, L, N = 8192, 4096, 8192

a = torch.randn(M, L, device="cuda", dtype=torch.float32)
b = torch.randn(L, N, device="cuda", dtype=torch.float32)
c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

a_ = from_dlpack(a)
b_ = from_dlpack(b)
c_ = from_dlpack(c)

naive_matmul = NaiveMatmul()
tiled_matmul = TiledMatmul(tile_width=12)

naive_matmul(a_, b_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5) 

c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
c_ = from_dlpack(c)
tiled_matmul(a_, b_, c_)
torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-3, rtol=1e-5) 
