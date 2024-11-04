from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_custom_ops',  # 自定义包名
    ext_modules=[
        CUDAExtension(
            'my_cuda_ops',  #扩展模块名称，不一定和name相同, 测试需要import my_cuda_ops
            ['./softmax/softmax.cu', './attention/attention.cu', './bindings.cpp'],  # 包含两个 CUDA 文件和一个 C++ 文件
            extra_compile_args={'nvcc': ['-arch=sm_70']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
