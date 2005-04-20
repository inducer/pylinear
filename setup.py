#!/usr/bin/env python

from distutils.core import setup,Extension
import glob
import os
import os.path

home = os.getenv("HOME")

boost_path = "%s/work/boost" % home
boost_ublas_bindings_path = "%s/work/boost-sandbox" % home
library_dirs = ["%s/pool/lib" % home] 

include_dirs = [boost_path, "algorithms"]
libraries = ["boost_python"]
extra_compile_args = ["-fmessage-length=0", "-Wno-sign-compare"]

#blas_libraries = ["f77blas", "atlas", "g2c"]
blas_libraries = ["blas"]

setup(name="PyLinear",
      version="0.92",
      description="Matrix handling in Python",
      author="Andreas Kloeckner",
      author_email="mathem@tiker.net",
      license = "BSD-Style",
      url="http://news.tiker.net/software/pylinear",
      packages=["pylinear"],
      ext_package="pylinear",
      ext_modules=[ Extension("_array", 
                              ["extensions/array.cpp", 
                               "extensions/vector.cpp", 
                               "extensions/matrix_dense.cpp",
                               "extensions/matrix_sparse_build.cpp",
                               "extensions/matrix_sparse_ex.cpp",
                               ],
                              include_dirs = include_dirs,
                              library_dirs = library_dirs,
                              libraries = libraries,
                              extra_compile_args = extra_compile_args,
                              ),
                    Extension( "_operation", 
                               ["extensions/operation.cpp", 
                                ],
          include_dirs = include_dirs + 
          [boost_ublas_bindings_path],
          library_dirs = library_dirs,
          libraries = libraries + ["umfpack", "amd", "arpack", "lapack"] + blas_libraries,
          extra_compile_args = extra_compile_args,
          ),
        ]
     )
