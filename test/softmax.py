import torch
import ctypes
import numpy as np
import torch.nn.functional as F
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def dataPrew(test_shape, test_axis):
    ndim = len(test_shape)
    dimsize = test_shape[test_axis]
    size = 1
    stride = 1
    for i in range(ndim - 1, -1, -1):
        size *= test_shape[i]
    for i in range(ndim - 1, -1, -1):
        
        if(test_axis == i):
            break
        stride *= test_shape[i]
    return size, stride, dimsize
def test(test_shape, test_axis, test_dtype, device):
    print(
        f"Testing Softmax on {device} with x_shape:{test_shape} , axis:{test_axis} ,dtype:{test_dtype}"
    )
    ndim = len(test_shape)
    size, stride, dimsize = dataPrew(test_shape, test_axis)
    Q = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)
    Q_output = torch.zeros(test_shape, device=device, dtype=torch.float32) 

    input_ptr = ctypes.cast(Q.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if test_dtype == torch.float32:
        if device == "cuda":
            torch_softmax_time = performance.CudaProfile((torch.softmax, (Q, test_axis)))  # 以毫秒为单位
            lib.softmax_nv_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_softmax_time = performance.CudaProfile((lib.softmax_nv_f32, (input_ptr, output_ptr, size, dimsize, stride)))  # 以毫秒为单位
        if device == "cpu":
            torch_softmax_time = performance.CpuProfile((torch.softmax, (Q, test_axis)))  # 以毫秒为单位
            lib.softmax_cpu_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_softmax_time = performance.CpuProfile((lib.softmax_cpu_f32, (input_ptr, output_ptr, size, dimsize, stride)))  # 以毫秒为单位
        if device == "mlu":
            torch_softmax_time = performance.BangProfile((torch.softmax, (Q, test_axis)))  # 以毫秒为单位
            
            '''
            frontsize = 1
            othersize = 1
            for s in range(ndim - 1, -1, -1):
                if (s < test_axis): 
                    frontsize *= test_shape[s]
                if (s != test_axis):
                    othersize *= test_shape[s];
            
            lib.softmax_bang_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_softmax_time = performance.BangProfile((lib.softmax_bang_f32, (input_ptr, output_ptr, othersize, dimsize, frontsize, stride, test_axis, ndim)))
            '''
            lib.softmax_cnnl_f32.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int)
            ]
            import numpy as np
            np_array = np.array(test_shape, dtype=np.int32)
            ctypes_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            custom_softmax_time = performance.BangProfile((lib.softmax_cnnl_f32, (input_ptr, output_ptr, ndim, test_axis, ctypes_array)))
    elif test_dtype == torch.float16:
        if device == "cuda":
            torch_softmax_time = performance.CudaProfile((torch.softmax, (Q, test_axis)))  # 以毫秒为单位
            '''
            lib.softmax_nv_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_softmax_time = performance.CudaProfile((lib.softmax_nv_f16, (input_ptr, output_ptr, size, dimsize, stride))) 
            '''
            import numpy as np
            np_array = np.array(test_shape, dtype=np.int32)
            ctypes_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            lib.softmax_cudnn_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int
            ]
            custom_softmax_time = performance.CudaProfile((lib.softmax_cudnn_f16, (input_ptr, output_ptr, ctypes_array, ndim)))
        if device == "cpu":
            torch_softmax_time = performance.CpuProfile((torch.softmax, (Q, test_axis)))  # 以毫秒为单位
            lib.softmax_cpu_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_softmax_time = performance.CpuProfile((lib.softmax_cpu_f16, (input_ptr, output_ptr, size, dimsize, stride)))  # 以毫秒为单位
        if device == "mlu":
            torch_softmax_time = performance.BangProfile((torch.softmax, (Q, test_axis)))  # 以毫秒为单位
            ndim = len(test_shape)
            '''
            frontsize = 1
            othersize = 1
            for s in range(ndim - 1, -1, -1):
                if (s < test_axis): 
                    frontsize *= test_shape[s]
                if (s != test_axis):
                    othersize *= test_shape[s];
            
            lib.softmax_bang_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_softmax_time = performance.BangProfile((lib.softmax_bang_f16, (input_ptr, output_ptr, othersize, dimsize, frontsize, stride, test_axis, ndim))) 
            '''
            lib.softmax_cnnl_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int)
            ]
            import numpy as np
            np_array = np.array(test_shape, dtype=np.int32)
            ctypes_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            custom_softmax_time = performance.BangProfile((lib.softmax_cnnl_f16, (input_ptr, output_ptr, ndim, test_axis, ctypes_array)))
    performance.logBenchmark(torch_softmax_time, custom_softmax_time)
    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = torch.softmax(Q, test_axis).to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # x_shape, axis
        ((700, 1200, 24), 0, torch.float32, 'cuda'),
        ((700, 1200, 24), 1, torch.float32, 'cuda'), 
        ((700, 1200, 24), 2, torch.float32, 'cuda'), 

        ((700, 1200, 24), 0, torch.float32, 'mlu'),
        ((700, 1200, 24), 1, torch.float32, 'mlu'), 
        ((700, 1200, 24), 2, torch.float32, 'mlu'), 

        ((70, 12, 24), 0, torch.float32, 'cpu'),
        ((70, 12, 24), 1, torch.float32, 'cpu'), 
        ((70, 12, 24), 2, torch.float32, 'cpu'), 

        # x_shape, axis
        ((700, 1200, 24), 0, torch.float16, 'cuda'),
        ((700, 1200, 24), 1, torch.float16, 'cuda'), 
        ((700, 1200, 24), 2, torch.float16, 'cuda'), 

        ((700, 1200, 24), 0, torch.float16, 'mlu'),
        ((700, 1200, 24), 1, torch.float16, 'mlu'), 
        ((700, 1200, 24), 2, torch.float16, 'mlu'), 

        ((70, 12, 24), 0, torch.float16, 'cpu'),
        ((70, 12, 24), 1, torch.float16, 'cpu'), 
        ((70, 12, 24), 2, torch.float16, 'cpu'), 
         
]
filtered_test_cases = [
    (test_shape, test_axis, test_dtype, device)
    for test_shape, test_axis, test_dtype, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for test_shape, test_axis, test_dtype, device in filtered_test_cases:
    test(test_shape, test_axis, test_dtype, device)