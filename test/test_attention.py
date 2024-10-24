import torch
import time
import numpy as np
import torch.nn.functional as F
# 添加上一层目录到模块搜索路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 现在可以导入 attentionCuda
import attentionCuda

def funAttention(Q, K, V): 
    return torch.softmax(Q@K.t(), dim = 1)@V
device = 'cuda' if torch.cuda.is_available() else 'cpu'

N, d = 1024, 1024  
print(f"Testing with N={N}, d={d}")

Q = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False) 
K = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)
V = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)
repeat = 20
#下面开始预热
for i in range(repeat):
    attTorch = funAttention(Q, K, V)
#正式计时
start_time = time.time()
for i in range(repeat):
    attTorch = funAttention(Q, K, V)
torch_flash_time = 1000 * (time.time() - start_time)
print("PyTorch Flash Attention time: %.6f ms"%(torch_flash_time / repeat))


# 创建输出张量
attHPC = torch.zeros([N, d], device = device, dtype = torch.float32)


#下面开始预热
for i in range(repeat):
    attentionCuda.attention(Q, K, V, N, d, attHPC)
#正式计时
start_time = time.time()
for i in range(repeat):
    attentionCuda.attention(Q, K, V, N, d, attHPC)
custom_attention_time = 1000 * (time.time() - start_time)
print("Cuda core Attention time: %.6f ms"%(custom_attention_time / repeat))

# 将结果转换回 PyTorch 张量以进行比较
tmpa = attTorch.to('cpu').numpy().reshape(-1,1).flatten()
tmpb = attHPC.to('cpu').numpy().reshape(-1,1).flatten()
atol = max(abs(tmpa - tmpb))

rtol = atol / max(abs(tmpb) + 1e-8)


print("absolute error:%.4e"%(atol))
print("relative error:%.4e"%(rtol))


# 对比性能
speedup = torch_flash_time / custom_attention_time
print(f"Speedup of Custom Tensor Core Attention over PyTorch Flash Attention: {speedup:.2f}x")
