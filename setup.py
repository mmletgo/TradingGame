"""
构建配置 - 用于编译 Cython 模块
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "src.market.orderbook.orderbook",
        ["src/market/orderbook/orderbook.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "src.market.account.position",
        ["src/market/account/position.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
