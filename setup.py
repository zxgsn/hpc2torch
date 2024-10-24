from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_custom_ops',#自定义包名
    ext_modules=[
        CUDAExtension(
            'softmaxCuda',#扩展模块名称，不一定和name相同
            ['./softmax/softmax.cu'],
            extra_compile_args={'nvcc': ['-arch=sm_70']}
        ),
        CUDAExtension(
            'attentionCuda',#扩展模块名称，不一定和name相同
            ['./attention/attention.cu'],
            extra_compile_args={'nvcc': ['-arch=sm_70']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

