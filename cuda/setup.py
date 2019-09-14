from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lstm_cuda',
    ext_modules=[
        CUDAExtension('lstm_cuda', [
            'lstm_cuda.cpp',
            'lstm_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
