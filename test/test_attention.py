import torch
import time
import numpy as np
import torch.nn.functional as F
import performance
# 添加上一层目录到模块搜索路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 现在可以导入 my_cuda_ops
import my_cuda_ops

def funAttention(Q, K, V): 
    return torch.softmax(Q@K.t(), dim = 1)@V
device = 'cuda' if torch.cuda.is_available() else 'cpu'

N, d = 1024, 1024  
print(f"Testing with N={N}, d={d}")

Q = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False) 
K = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)
V = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)

torch_flash_time = performance.CudaProfile((funAttention, (Q, K, V)))



# 创建输出张量
attHPC = torch.zeros([N, d], device = device, dtype = torch.float32)


custom_attention_time = performance.CudaProfile((my_cuda_ops.attention, (Q, K, V, N, d, attHPC)))
performance.logBenchmark(torch_flash_time, custom_attention_time)

# 将结果转换回 PyTorch 张量以进行比较
tmpa = funAttention(Q, K, V).to('cpu').numpy().reshape(-1,1).flatten()
my_cuda_ops.attention(Q, K, V, N, d, attHPC)
tmpb = attHPC.to('cpu').numpy().reshape(-1,1).flatten()
atol = max(abs(tmpa - tmpb))

rtol = atol / max(abs(tmpb) + 1e-8)


print("absolute error:%.4e"%(atol))
print("relative error:%.4e"%(rtol))


# 对比性能
speedup = torch_flash_time / custom_attention_time
print(f"Speedup of Custom Tensor Core Attention over PyTorch Flash Attention: {speedup:.2f}x")
