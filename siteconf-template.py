# -------------------------------------------------------------------------------------------
# User servicable part
# -------------------------------------------------------------------------------------------

HAVE_BLAS = False
HAVE_LAPACK = False
HAVE_ARPACK = False
HAVE_UMFPACK = False

BOOST_INCLUDE_DIRS = ["/home/ak/work/boost"]
BOOST_UBLAS_BINDINGS_INCLUDE_DIRS = ["/home/ak/work/boost-sandbox"]
BOOST_LIBRARY_DIRS = ["/home/ak/pool/lib"] 
BPL_LIBRARIES = ["boost_python"]

BLAS_LIBRARY_DIRS = []
BLAS_LIBRARIES = ["blas"]

LAPACK_LIBRARY_DIRS = []
LAPACK_LIBRARIES = ["lapack"]

ARPACK_LIBRARY_DIRS = []
ARPACK_LIBRARIES = ["arpack"]

# omit the last "umfpack/"
UMFPACK_INCLUDE_DIRS = [] 
UMFPACK_LIBRARY_DIRS = []
UMFPACK_LIBRARIES = ["umfpack", "amd"]

EXTRA_COMPILE_ARGS = ["-fmessage-length=0", "-Wno-sign-compare"]

