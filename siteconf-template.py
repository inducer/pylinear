# --------------------------------------------------------------------
# Specify your configuration below.
# See documentation for hints.
# --------------------------------------------------------------------

# Change to "True" (without the quotes) if the respective package is available.
HAVE_BLAS = False
HAVE_LAPACK = False
HAVE_ARPACK = False
HAVE_UMFPACK = False

BOOST_INCLUDE_DIRS = []
BOOST_LIBRARY_DIRS = [] 
BPL_LIBRARIES = ["boost_python"]

BOOST_UBLAS_BINDINGS_INCLUDE_DIRS = ["/home/andreas/work/boost-sandbox"]

BLAS_LIBRARY_DIRS = []
BLAS_LIBRARIES = ["blas"]

LAPACK_LIBRARY_DIRS = []
LAPACK_LIBRARIES = ["lapack"]

ARPACK_LIBRARY_DIRS = []
ARPACK_LIBRARIES = ["arpack"]

# omit the last "umfpack/" on the include path
UMFPACK_INCLUDE_DIRS = [] 
UMFPACK_LIBRARY_DIRS = []
UMFPACK_LIBRARIES = ["umfpack", "amd"]

#EXTRA_COMPILE_ARGS = ["-fmessage-length=0", "-Wno-sign-compare"]
EXTRA_COMPILE_ARGS = ["-Wno-sign-compare"]

