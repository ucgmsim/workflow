from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "workflow.scripts.merge_ts_loop",
        ["workflow/scripts/merge_ts_loop.pyx"],
    ),
]

setup(
    name="workflow",
    ext_modules=cythonize(extensions),
)
