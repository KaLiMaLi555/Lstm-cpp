from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lstm_cpp',
      ext_modules=[cpp_extension.CppExtension('lstm_cpp', ['lstm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})