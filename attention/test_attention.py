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
    flash_attention_output = funAttention(Q, K, V)
torch_flash_time = time.time() - start_time
print("PyTorch Flash Attention time: %.6f seconds"%(torch_flash_time / repeat))


# 创建输出张量
custom_output = np.zeros((N, d), dtype=np.float32)  

# 将输入张量转换为 numpy 并确保是 float32 类型
Q_np = Q.squeeze(0).cpu().numpy().astype(np.float32)
K_np = K.squeeze(0).cpu().numpy().astype(np.float32)
V_np = V.squeeze(0).cpu().numpy().astype(np.float32)

start_time = time.time()
for i in range(repeat):
    attention.attention(Q_np, K_np, V_np, N, d, custom_output)
custom_attention_time = time.time() - start_time
print("Custom Tensor Core Attention time: %.6f seconds"%(custom_attention_time / repeat))

# 将结果转换回 PyTorch 张量以进行比较
custom_output_tensor = torch.tensor(custom_output, device=device)

absolute_error = torch.abs(flash_attention_output - custom_output_tensor)
relative_error = absolute_error / (torch.abs(flash_attention_output) + 1e-8)  # 防止除零

print(f"Absolute error (mean): {torch.mean(absolute_error).item()}")
print(f"Absolute error (max): {torch.max(absolute_error).item()}")
print(f"Relative error (mean): {torch.mean(relative_error).item()}")
print(f"Relative error (max): {torch.max(relative_error).item()}")

# 对比性能
speedup = torch_flash_time / custom_attention_time
print(f"Speedup of Custom Tensor Core Attention over PyTorch Flash Attention: {speedup:.2f}x")
