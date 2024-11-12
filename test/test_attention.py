import torch
import ctypes
import numpy as np
import torch.nn.functional as F
import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_cuda_library.so')
lib = ctypes.CDLL(lib_path)
lib.attention_nv_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)
]
# 定义函数参数类型
def funAttention(Q, K, V): 
    return torch.softmax(Q@K.t(), dim = 1)@V
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

N, d = 1024, 1024  
print(f"Testing with N={N}, d={d}")

Q = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False) 
K = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)
V = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)

torch_flash_time = performance.CudaProfile((funAttention, (Q, K, V)))



# 创建输出张量
attHPC = torch.zeros([N, d], device = device, dtype = torch.float32)

Q_ptr = ctypes.cast(Q.data_ptr(), ctypes.POINTER(ctypes.c_float))
K_ptr = ctypes.cast(K.data_ptr(), ctypes.POINTER(ctypes.c_float))
V_ptr = ctypes.cast(V.data_ptr(), ctypes.POINTER(ctypes.c_float))
attHPC_ptr = ctypes.cast(attHPC.data_ptr(), ctypes.POINTER(ctypes.c_float))
# 假设 N 和 d 是整数
N = ctypes.c_int(N)
d = ctypes.c_int(d)

# 调用 C 函数
custom_attention_time = performance.CudaProfile((
    lib.attention_nv_f32,
    (Q_ptr, K_ptr, V_ptr, N, d, attHPC_ptr)
))
performance.logBenchmark(torch_flash_time, custom_attention_time)

# 将结果转换回 PyTorch 张量以进行比较
tmpa = funAttention(Q, K, V).to('cpu').numpy().reshape(-1,1).flatten()
lib.attention_nv_f32(Q_ptr, K_ptr, V_ptr, N, d, attHPC_ptr)
tmpb = attHPC.to('cpu').numpy().reshape(-1,1).flatten()
atol = max(abs(tmpa - tmpb))

rtol = atol / max(abs(tmpb) + 1e-8)


print("absolute error:%.4e"%(atol))
print("relative error:%.4e"%(rtol))


# 对比性能
speedup = torch_flash_time / custom_attention_time
print(f"Speedup of Custom Tensor Core Attention over PyTorch Flash Attention: {speedup:.2f}x")
