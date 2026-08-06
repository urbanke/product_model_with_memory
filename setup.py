from setuptools import Extension, setup

import numpy


setup(
    ext_modules=[
        Extension(
            "product_model_with_memory._graphical_margin_c",
            ["src/product_model_with_memory/_graphical_margin_c.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3"],
        )
    ]
)
