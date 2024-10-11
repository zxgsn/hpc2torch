from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax',#指定python包的名称是softmax
    ext_modules=[
        CUDAExtension(
            'softmax',#扩展模块名称，不一定和name相同
            ['softmax.cu'],
            extra_compile_args={'nvcc': ['-arch=sm_70']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

