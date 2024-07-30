from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "merge_ts.merge_ts_loop",
        ["merge_ts/merge_ts_loop.pyx"],
    ),
]

setup(
    name="workflow",
    ext_modules=cythonize(extensions),
)
