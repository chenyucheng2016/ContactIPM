"""
Build script for ContactIPM pybind11 bindings.

Usage:
    cd python
    pip install -e .

Or:
    python setup.py build_ext --inplace
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class get_pybind_include:
    """Helper class to determine the pybind11 include path."""
    def __str__(self):
        return pybind11.get_include()


# Project root (parent of python/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Include directories
include_dirs = [
    get_pybind_include(),
    os.path.join(project_root, 'include'),
]

# Source files
sources = ['bindings.cpp']

# Compiler flags
if sys.platform == 'win32':
    compile_args = ['/O2', '/std:c++17', '/utf-8']
else:
    compile_args = ['-O3', '-std=c++17', '-fvisibility=hidden']

# Extension module
ext_modules = [
    Extension(
        'contactipm',
        sources=sources,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=compile_args,
    ),
]


class BuildExt(build_ext):
    """Custom build extension for pybind11."""
    def build_extensions(self):
        # Check compiler
        if sys.platform == 'win32':
            # MSVC
            for ext in self.extensions:
                ext.extra_compile_args.append('/EHsc')
        else:
            # GCC/Clang
            for ext in self.extensions:
                ext.extra_compile_args.append('-Wall')
        
        build_ext.build_extensions(self)


setup(
    name='contactipm',
    version='0.1.0',
    author='ContactIPM Team',
    description='Python bindings for ContactIPM NMPC solver',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6'],
    setup_requires=['pybind11>=2.6'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.8',
)
