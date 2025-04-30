from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='s4d_cuda',
    ext_modules=[
        CUDAExtension(
            name='s4d_cuda_kernel',
            sources=[
                'src/s4dkernel_wrapper.cpp',
                'src/s4dkernel_kernel.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    version="1.0.0"
)
