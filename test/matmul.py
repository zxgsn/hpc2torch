import torch
import ctypes
import torch.nn.functional as F
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

# 定义函数参数类型
def funAttention(Q, K, V): 
    return torch.softmax(Q@K.t(), dim = 1)@V

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)
def test(M, K, N, test_dtype, device):
    print(
        f"Testing Attention on {device} with M-K-N:{M, K, N} , dtype:{test_dtype}"
    )
    A = torch.randn([M, K], device=device, dtype=torch.float32, requires_grad=False) 
    B = torch.randn([K, N], device=device, dtype=torch.float32, requires_grad=False)
    C = torch.zeros([M, N], device=device, dtype=torch.float32, requires_grad=False)
    

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if device == "cuda":
        torch_matmul_time = performance.CudaProfile((torch.matmul, (A, B)))
        
        lib.matmul_cuda_f32.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
        ]
        custom_matmul_time = performance.CudaProfile((
            lib.matmul_cuda_f32,
            (A_ptr, B_ptr, C_ptr, M, K, N)
        ))
        '''
        lib.matmul_cudnn_f32.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
        ]
        custom_matmul_time = performance.CudaProfile((
            lib.matmul_cudnn_f32,
            (A_ptr, B_ptr, C_ptr, M, K, N)
        ))
        '''
        
        
    performance.logBenchmark(torch_matmul_time, custom_matmul_time)
    tmpa = torch.matmul(A, B).to('cpu').numpy().flatten()
    tmpb = C.to('cpu').numpy().flatten()
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # M, K, N, test_dtype, device
        (1024, 128, 1024, torch.float32, 'cuda'),
        (1024, 256, 1024, torch.float32, 'cuda'),
        (1024, 512, 1024, torch.float32, 'cuda'),
        (1024, 1024, 1024, torch.float32, 'cuda'),
]
filtered_test_cases = [
    (M, K, N, test_dtype, device)
    for M, K, N, test_dtype, device in test_cases
    if device == args.device
]

for M, K, N, test_dtype, device in filtered_test_cases:
    test(M, K, N, test_dtype, device)