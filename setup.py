#!/usr/bin/env python

from distutils.core import setup,Extension
import glob
import os
import os.path

home = os.getenv("HOME")
boost_path = "%s/work/boost" % home
include_dirs = [boost_path]
library_dirs = ["%s/pool/lib" % home] 
libraries = ["boost_python"]
extra_compile_args = ["-fmessage-length=0"]

setup(name="PyLinear",
      version="0.90",
      description="Matrix handling in Python",
      author="Andreas Kloeckner",
      author_email="ak@ixion.net",
      license = "GNU GPL",
      url="http://pylinear.sf.net",
      packages=["pylinear"],
      ext_package="pylinear",
      ext_modules=[
        Extension(
          "matrices_internal", 
          ["extensions/matrices.cpp", "extensions/matrices2.cpp", "extensions/matrices3.cpp"],
          include_dirs = include_dirs,
          library_dirs = library_dirs,
          libraries = libraries,
          extra_compile_args = extra_compile_args,
          ),
        ]
     )
