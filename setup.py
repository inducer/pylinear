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
extra_compile_args = ["-fmessage-length=0"]

#blas_libraries = ["f77blas", "atlas", "g2c"]
blas_libraries = ["blas2"]

setup(name="PyLinear",
      version="0.91",
      description="Matrix handling in Python",
      author="Andreas Kloeckner",
      author_email="ak@ixion.net",
      license = "BSD-Style",
      url="http://pylinear.sf.net",
      packages=["pylinear"],
      ext_package="pylinear",
      ext_modules=[ Extension("_matrices", 
                              ["extensions/matrices.cpp", 
                               "extensions/vectors.cpp", 
                               "extensions/matrices_sparse_build.cpp",
                               "extensions/matrices_sparse_ex.cpp",
                               #"extensions/matrices_sym_ex.cpp", 
                               #"extensions/matrices_sym_build.cpp",
                               ],
                              include_dirs = include_dirs,
                              library_dirs = library_dirs,
                              libraries = libraries,
                              extra_compile_args = extra_compile_args,
                              ),
                    Extension( "algorithms/", 
                               ["extensions/algorithms.cpp", 
                                ],
          include_dirs = include_dirs + 
          [boost_ublas_bindings_path, "3rdparty/arpack"],
          library_dirs = library_dirs,
          libraries = libraries + ["umfpack", "amd", "arpack", "lapack"] + blas_libraries,
          extra_compile_args = extra_compile_args,
          ),
        ]
     )
