#!/usr/bin/env python

from distutils.core import setup,Extension
import glob
import os
import os.path
import sys

try:
    execfile("siteconf.py")
except IOError:
    print "*** Please copy siteconf-template.py to siteconf.py,"
    print "*** then edit siteconf.py to match your environment."
    sys.exit(1)

# These are in Fortran. No headers available.
BLAS_INCLUDE_DIRS = []
LAPACK_INCLUDE_DIRS = []
ARPACK_INCLUDE_DIRS = []

INCLUDE_DIRS = ["algorithms"] + \
               BOOST_INCLUDE_DIRS
LIBRARY_DIRS = BOOST_LIBRARY_DIRS
LIBRARIES = BPL_LIBRARIES

OP_EXTRA_INCLUDE_DIRS = BOOST_UBLAS_BINDINGS_INCLUDE_DIRS
OP_EXTRA_LIBRARY_DIRS = []
OP_EXTRA_LIBRARIES = []

USE_BLAS = HAVE_BLAS
USE_LAPACK = HAVE_LAPACK and HAVE_BLAS
USE_ARPACK = HAVE_ARPACK and USE_LAPACK
USE_UMFPACK = HAVE_BLAS and HAVE_BLAS

if HAVE_LAPACK and not HAVE_BLAS:
    print "*** LAPACK disabled because BLAS is missing"
if HAVE_ARPACK and not USE_LAPACK:
    print "*** ARPACK disabled because LAPACK is not usable/missing"
if HAVE_UMFPACK and not HAVE_BLAS:
    print "*** UMFPACK disabled because BLAS is missing"

DEFINES = {}

def handle_component(comp):
    if globals()["USE_"+comp]:
        globals()["DEFINES"]["USE_"+comp] = 1
        globals()["OP_EXTRA_INCLUDE_DIRS"] += globals()[comp+"_INCLUDE_DIRS"]
        globals()["OP_EXTRA_LIBRARY_DIRS"] += globals()[comp+"_LIBRARY_DIRS"]
        globals()["OP_EXTRA_LIBRARIES"] += globals()[comp+"_LIBRARIES"]

handle_component("BLAS")
handle_component("LAPACK")
handle_component("ARPACK")
handle_component("UMFPACK")

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
                              include_dirs = INCLUDE_DIRS,
                              library_dirs = LIBRARY_DIRS,
                              libraries = LIBRARIES,
                              extra_compile_args = EXTRA_COMPILE_ARGS,
                              ),
                    Extension( "_operation", 
                               ["extensions/operation.cpp",
                                ],
                               include_dirs = INCLUDE_DIRS + OP_EXTRA_INCLUDE_DIRS,
                               library_dirs = LIBRARY_DIRS + OP_EXTRA_LIBRARY_DIRS,
                               libraries = LIBRARIES + OP_EXTRA_LIBRARIES,
                               extra_compile_args = EXTRA_COMPILE_ARGS,
          ),
        ]
     )
