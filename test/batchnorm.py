import torch
import ctypes
import torch.nn.functional as F
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)
def get_mean_variance(x, dtype):
    dims = tuple(range(x.ndim))
    reduction_dims = tuple(d for d in dims if d != 1)  # Exclude the channel dimension
    return x.mean(dim=reduction_dims, dtype=dtype), x.var(
        dim=reduction_dims, unbiased=False
    ).to(dtype)

def batch_norm(x, scale, b, mean, var, eps):
    ndim = len(x.shape)
    if ndim <= 1 or ndim > 5:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None
    return F.batch_norm(x, mean, var, scale, b, training=False, eps=eps)


def test(test_shape, test_dtype, eps, device):
    print(
        f"Testing Batchnorm on {device} with test_shape:{test_shape}, dtype:{test_dtype}, eps:{eps}"
    )
    ndim = len(test_shape)
    cSize = test_shape[1]
    input = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)
    #cnnlBatchnorm支持scale类型f16但是input类型f32的情况，但是手写batchnorm必须保证input,scale数据类型保持一致
    bn_dtype = test_dtype if test_dtype != torch.float16 else torch.float32
    scale = torch.rand(cSize, device=device, dtype=bn_dtype, requires_grad=False)
    bias = torch.rand(cSize, device=device, dtype=bn_dtype, requires_grad=False)
    mean, var = get_mean_variance(input, bn_dtype)
    
    output = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    scale_ptr = ctypes.cast(scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    mean_ptr = ctypes.cast(mean.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    var_ptr = ctypes.cast(var.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    import numpy as np
    np_array = np.array(test_shape, dtype=np.int32)
    shape = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    
    if test_dtype == torch.float32:
        
        if device == "mlu":
            torch_batchnorm_time = performance.BangProfile((batch_norm, (input, scale, bias, mean, var, eps)))  # 以毫秒为单位
            '''
            lib.batchnorm_cnnl_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_float
            ]           
            custom_batchnorm_time = \
            performance.BangProfile((lib.batchnorm_cnnl_f32, (input_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, output_ptr, shape, ndim, eps)))
            '''
            lib.batchnorm_bang_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_float
            ]           
            custom_batchnorm_time = \
            performance.BangProfile((lib.batchnorm_bang_f32, (input_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, output_ptr, shape, ndim, eps)))
            
    if test_dtype == torch.float16:
        
        if device == "mlu":
            torch_batchnorm_time = performance.BangProfile((batch_norm, (input, scale, bias, mean, var, eps)))  # 以毫秒为单位
            
            lib.batchnorm_cnnl_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_float
            ]
            custom_batchnorm_time = \
            performance.BangProfile((lib.batchnorm_cnnl_f16, (input_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, output_ptr, shape, ndim,eps)))
            '''
            lib.batchnorm_bang_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_float
            ]           
            custom_batchnorm_time = \
            performance.BangProfile((lib.batchnorm_bang_f16, (input_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, output_ptr, shape, ndim, eps)))
            '''
    performance.logBenchmark(torch_batchnorm_time, custom_batchnorm_time)

    # 将结果转换回 PyTorch 张量以进行比较
    
    tmpa = batch_norm(input, scale, bias, mean, var, eps).to('cpu').detach().numpy().flatten()
    
    tmpb = output.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test batchnorm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [       
        ((700, 12), torch.float32, 1e-5, 'mlu'), 
        ((700, 12, 24), torch.float32, 1e-5, 'mlu'),
        ((700, 12, 24, 32), torch.float32, 1e-5, 'mlu'),
        ((700, 12, 24, 32, 64), torch.float32, 1e-5, 'mlu'),

        ((700, 12), torch.float16, 1e-5, 'mlu'),
        ((700, 12, 24), torch.float16, 1e-5, 'mlu'),
        ((700, 12, 24, 32), torch.float16, 1e-5, 'mlu'),
        ((700, 12, 24, 32, 64), torch.float16, 1e-5, 'mlu'),        
]

filtered_test_cases = [
    (test_shape, test_dtype, eps, device)
    for test_shape, test_dtype, eps, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for test_shape, test_dtype, eps, device in filtered_test_cases:
    test(test_shape, test_dtype, eps, device)