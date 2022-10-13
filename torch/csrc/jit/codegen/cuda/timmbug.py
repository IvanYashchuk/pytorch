import torch
from torch._C._nvfuser import FusionDefinition, Fusion, DataType

def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True], dtype=DataType.Float)
    T1 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
    T2 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True], dtype=DataType.Float)
    T3 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
    T4 = fd.define_tensor(symbolic_sizes=[-1, -1, -1, -1], contiguous=[True, True, True, True], dtype=DataType.Float)
    T5 = fd.ops.broadcast_in_dim(T0, output_shape=[32, 56, 56, 1], broadcast_dims=[0, 1, 2])
    T6 = fd.ops.broadcast_in_dim(T1, output_shape=[32, 56, 56, 128], broadcast_dims=[3])
    T7 = fd.ops.broadcast_in_dim(T2, output_shape=[32, 56, 56, 1], broadcast_dims=[0, 1, 2])
    T8 = fd.ops.broadcast_in_dim(T3, output_shape=[32, 56, 56, 128], broadcast_dims=[3])
    T9 = fd.ops.broadcast_in_dim(T5, output_shape=[32, 56, 56, 128], broadcast_dims=[0, 1, 2, 3])
    S10 = fd.define_constant(1.00000e-06)
    T11 = fd.ops.add(T7, S10)
    T12 = fd.ops.sub(T4, T9)
    T13 = fd.ops.rsqrt(T11)
    T14 = fd.ops.broadcast_in_dim(T13, output_shape=[32, 56, 56, 128], broadcast_dims=[0, 1, 2, 3])
    T15 = fd.ops.mul(T12, T14)
    T16 = fd.ops.mul(T15, T6)
    T17 = fd.ops.add(T16, T8)
    T18 = fd.ops.cast(T17, dtype=DataType.Float)
    fd.add_output(T5)
    fd.add_output(T13)
    fd.add_output(T18)

inputs = [
    torch.randn(32, 56, 56, device='cuda'),
    torch.randn(128, device='cuda'),
    torch.randn(32, 56, 56, device='cuda'),
    torch.randn(128, device='cuda'),
    torch.randn(32, 56, 56, 128, device='cuda'),
]

fs = Fusion()
with FusionDefinition(fs) as fd:
    nvfuser_fusion_id1(fd)

for _ in range(5) :
    fs.execute(inputs)
