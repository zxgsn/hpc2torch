from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention',
    ext_modules=[
        CUDAExtension(
            'attention',
            ['attention.cu'],
            extra_compile_args={'nvcc': ['-arch=sm_70']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
