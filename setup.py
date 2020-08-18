from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

setup(name='span_prune_cpp',
      ext_modules=[cpp_extension.CppExtension('span_prune_cpp', ['span_prune.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

