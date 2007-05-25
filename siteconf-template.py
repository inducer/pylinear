# --------------------------------------------------------------------
# Specify your configuration below.
# See documentation for hints.
# --------------------------------------------------------------------

PYLINEAR_CONF_TEMPLATE_VERSION = 2

# Change to "True" (without the quotes) if the respective package is available.
HAVE_BLAS = False
HAVE_LAPACK = False
HAVE_ARPACK = False
HAVE_UMFPACK = False

# DASKR is included in PyLinear's source for your convenience.
# You may go to fortran/daskr and run ./build.sh, then set 
# this variable to True. The default settings below should then
# suffice to include DASKR in your build.

HAVE_DASKR = False

# --------------------------------------------------------------------
# Path options
# --------------------------------------------------------------------

BOOST_INCLUDE_DIRS = []
BOOST_LIBRARY_DIRS = [] 
BPL_LIBRARIES = ["boost_python-mt"]

BOOST_UBLAS_BINDINGS_INCLUDE_DIRS = ["/home/andreas/work/boost-sandbox"]

BLAS_LIBRARY_DIRS = []
BLAS_LIBRARIES = ["blas"]

LAPACK_LIBRARY_DIRS = []
LAPACK_LIBRARIES = ["lapack"]

ARPACK_LIBRARY_DIRS = []
ARPACK_LIBRARIES = ["arpack"]

# *** CHANGE with respect to prior versions: 
# include trailing "umfpack" or "ufsparse"
# make sure you use boost-bindings release 2006-04-30 or newer.
# older version, stand-alone
#UMFPACK_INCLUDE_DIRS = ["/usr/include/umfpack"] 
# newer version, stand-alone
UMFPACK_INCLUDE_DIRS = ["/usr/include/ufsparse"] 
UMFPACK_LIBRARY_DIRS = []
UMFPACK_LIBRARIES = ["umfpack", "amd"]

DASKR_LIBRARY_DIRS = ["fortran/daskr"]
DASKR_LIBRARIES = ["daskr"]

# --------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------
#EXTRA_COMPILE_ARGS = ["-fmessage-length=0", "-Wno-sign-compare"]
EXTRA_COMPILE_ARGS = ["-Wno-sign-compare"]

