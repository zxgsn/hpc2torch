import torch
import time
import attention  
import numpy as np
import torch.nn.functional as F
def funAttention(Q, K, V): 
    return torch.softmax(Q@K.t(), dim = 1)@V
device = 'cuda' if torch.cuda.is_available() else 'cpu'

N, d = 1024, 1024  
print(f"Testing with N={N}, d={d}")

Q = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False) 
K = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)
V = torch.randn(N, d, device=device, dtype=torch.float32, requires_grad=False)
repeat = 20
start_time = time.time()
for i in range(repeat):
    attTorch = funAttention(Q, K, V)
torch_flash_time = 1000 * (time.time() - start_time)
print("PyTorch Flash Attention time: %.6f ms"%(torch_flash_time / repeat))


# 创建输出张量
attHPC = np.zeros((N, d), dtype=np.float32)  

# 将输入张量转换为 numpy 并确保是 float32 类型
Q_np = Q.cpu().numpy().astype(np.float32)
K_np = K.cpu().numpy().astype(np.float32)
V_np = V.cpu().numpy().astype(np.float32)

start_time = time.time()
for i in range(repeat):
    attention.attention(Q_np, K_np, V_np, N, d, attHPC)
custom_attention_time = 1000 * (time.time() - start_time)
print("Cuda core Attention time: %.6f ms"%(custom_attention_time / repeat))

# 将结果转换回 PyTorch 张量以进行比较
tmpa = attTorch.to('cpu').reshape(-1,1)
tmpb = attHPC.reshape(-1,1)
atol = max(abs(tmpa - tmpb))
rtol = atol / max(abs(tmpb) + 1e-8)


print("absolute error:%.4e"%(atol))
print("relative error:%.4e"%(rtol))

# 对比性能
speedup = torch_flash_time / custom_attention_time
print(f"Speedup of Custom Tensor Core Attention over PyTorch Flash Attention: {speedup:.2f}x")
