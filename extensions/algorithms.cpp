#include <boost/python.hpp>
#include <cg.h>
#include <umfpack.h>
#include <arpack.h>
#include "meta.h"



// shape accessors ------------------------------------------------------------
template <typename MatrixType>
inline python::object getShape(const MatrixType &m)
{ 
  return python::make_tuple(m.size1(), m.size2());
}




// wrappers -------------------------------------------------------------------
template <typename ValueType>
class matrix_operator_wrapper : public matrix_operator<ValueType>
{
    PyObject *m_self;

  public:
    typedef 
      typename matrix_operator<ValueType>::vector_type
      vector_type;

    matrix_operator_wrapper(PyObject *self, const matrix_operator<ValueType> &)
    : m_self(self)
    { 
    }

    unsigned size1() const
    {
      return python::extract<unsigned>(python::call_method<python::tuple>(m_self, "shape")[0]);
    }
    unsigned size2() const
    {
      return python::extract<unsigned>(python::call_method<python::tuple>(m_self, "shape")[1]);
    }
    void apply(const vector_type &before, vector_type &after) const
    {
      return python::call_method<void>(m_self, "apply", before, after);
    }
};




template <typename ValueType>
class algorithm_matrix_operator_wrapper : public algorithm_matrix_operator<ValueType>
{
    PyObject *m_self;

  public:
    typedef 
      typename matrix_operator<ValueType>::vector_type
      vector_type;

    algorithm_matrix_operator_wrapper(PyObject *self, const algorithm_matrix_operator<ValueType> &)
    : m_self(self)
    { 
    }

    unsigned size1() const
    {
      return python::extract<unsigned>(python::call_method<python::tuple>(m_self, "shape")[0]);
    }
    unsigned size2() const
    {
      return python::extract<unsigned>(python::call_method<python::tuple>(m_self, "shape")[1]);
    }
    void apply(const vector_type &before, vector_type &after) const
    {
      return python::call_method<void>(m_self, "apply", before, after);
    }
};




template <typename ValueType>
class iterative_solver_matrix_operator_wrapper : public iterative_solver_matrix_operator<ValueType>
{
    PyObject *m_self;

  public:
    typedef 
      typename matrix_operator<ValueType>::vector_type
      vector_type;

    iterative_solver_matrix_operator_wrapper(PyObject *self, const iterative_solver_matrix_operator<ValueType> &)
    : m_self(self)
    { 
    }

    unsigned size1() const
    {
      return python::extract<unsigned>(python::call_method<python::tuple>(m_self, "shape")[0]);
    }
    unsigned size2() const
    {
      return python::extract<unsigned>(python::call_method<python::tuple>(m_self, "shape")[1]);
    }
    void apply(const vector_type &before, vector_type &after) const
    {
      return python::call_method<void>(m_self, "apply", before, after);
    }
};




// ublas_matrix_operator ------------------------------------------------------
template <typename MatrixType>
static ublas_matrix_operator<MatrixType> *makeMatrixOperator(const MatrixType &mat)
{
  return new ublas_matrix_operator<MatrixType>(mat);
}




struct ublas_matrix_operator_exposer
{
  template <typename MatrixType>
  void expose(const std::string &python_mattype, MatrixType) const
  {
    typedef 
      typename MatrixType::value_type
      value_type;
    typedef 
      typename matrix_operator<value_type>::vector_type
      vector_type;

    python::class_<ublas_matrix_operator<MatrixType>, 
    python::bases<matrix_operator<value_type> > >
      (("MatrixOperator" + python_mattype).c_str(),
       python::init<const MatrixType &>()[python::with_custodian_and_ward<1,2>()]);
    python::def("makeMatrixOperator", makeMatrixOperator<MatrixType>,
        python::return_value_policy<
        python::manage_new_object,
        python::with_custodian_and_ward_postcall<0, 1> >());
  }
};




// umfpack_matrix_operator ----------------------------------------------------
template <typename PythonClass>
struct umfpack_matrix_operator_constructor_exposer
{
  PythonClass &m_pyclass;

public:
  umfpack_matrix_operator_constructor_exposer(PythonClass &pyclass)
  : m_pyclass(pyclass)
  {
  }

  template <typename MatrixType>
  void expose(const std::string &python_mattype, MatrixType) const
  {
    m_pyclass
      .def(python::init<const MatrixType &>());
  }
};





// generic instantiation infrastructure ---------------------------------------
template <typename Exposer, typename ValueType>
static void exposeForAllSimpleTypes(const std::string &python_eltname, const Exposer &exposer, ValueType)
{
  exposer.expose("Matrix" + python_eltname, ublas::matrix<ValueType>());
  exposer.expose("SparseExecuteMatrix" + python_eltname, ublas::compressed_matrix<ValueType>());
  exposer.expose("SparseBuildMatrix" + python_eltname, ublas::coordinate_matrix<ValueType>());
}




template <typename Exposer>
static void exposeForAllMatrices(const Exposer &exposer, double)
{
  exposeForAllSimpleTypes("Float64", exposer, double());

  exposer.expose("SparseSymmetricExecuteMatrixFloat64", managed_symmetric_adaptor<
      ublas::compressed_matrix<double> >());
  exposer.expose("SparseSymmetricBuildMatrixFloat64", managed_symmetric_adaptor<
      ublas::coordinate_matrix<double> >());
}




template <typename Exposer>
static void exposeForAllMatrices(const Exposer &exposer, std::complex<double>)
{
  exposeForAllSimpleTypes("Complex64", exposer, std::complex<double>());

  exposer.expose("SparseHermitianExecuteMatrixComplex64", managed_hermitian_adaptor<
      ublas::compressed_matrix<std::complex<double> > >());
  exposer.expose("SparseHermitianBuildMatrixComplex64", managed_hermitian_adaptor<
      ublas::coordinate_matrix<std::complex<double> > >());
}




template <typename Exposer>
static void exposeForAllMatrices(const Exposer &exposer)
{
  exposeForAllMatrices(exposer, double());
  exposeForAllMatrices(exposer, std::complex<double>());
}




// matrix operators -----------------------------------------------------------
template <typename ValueType>
static void exposeMatrixOperators(const std::string &python_eltname, ValueType)
{
  {
    typedef matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, matrix_operator_wrapper<ValueType>, noncopyable >
      (("MatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("shape", &getShape<wrapped_type>)
      .def("typecode", &typecode<wrapped_type>)
      .def("apply", &wrapped_type::apply)
      ;
  }

  {
    typedef algorithm_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
    python::bases<matrix_operator<ValueType> >,
    algorithm_matrix_operator_wrapper<ValueType>, noncopyable >
      (("AlgorithmMatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("debug_level", &wrapped_type::getDebugLevel, &wrapped_type::setDebugLevel)
      .add_property("last_iteration_count", &wrapped_type::getLastIterationCount)
      ;
  }

  {
    typedef iterative_solver_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
    python::bases<algorithm_matrix_operator<ValueType> >,
    iterative_solver_matrix_operator_wrapper<ValueType>, noncopyable >
      (("IterativeSolverMatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("max_iterations", &wrapped_type::getMaxIterations, &wrapped_type::setMaxIterations)
      .add_property("tolerance", &wrapped_type::getTolerance, &wrapped_type::setTolerance)
      ;
  }

  {
    typedef identity_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
    python::bases<matrix_operator<ValueType> > >
      (("IdentityMatrixOperator"+python_eltname).c_str(), 
       python::init<unsigned>());
  }

  {
    typedef composite_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
    python::bases<matrix_operator<ValueType> > >
      (("CompositeMatrixOperator"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<ValueType> &, 
         const matrix_operator<ValueType> &>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  {
    typedef cg::cg_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
    python::bases<iterative_solver_matrix_operator<ValueType> > >
      (("CGMatrixOperator"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<ValueType> &, 
         const matrix_operator<ValueType>&, 
         unsigned, double>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  { typedef umfpack::umfpack_matrix_operator<ValueType> wrapped_type;
    typedef 
      python::class_<wrapped_type, 
      python::bases<algorithm_matrix_operator<ValueType> >, boost::noncopyable>    
        wrapper_type;

    wrapper_type pyclass(("UMFPACKMatrixOperator"+python_eltname).c_str(), 
       python::init<const ublas::identity_matrix<ValueType> &>());
    
    exposeForAllMatrices(
        umfpack_matrix_operator_constructor_exposer<wrapper_type>(pyclass), 
        ValueType());
  }
}




// arpack ---------------------------------------------------------------------
template <typename ResultsType>
static typename ResultsType::value_container::iterator beginRitzValues(ResultsType &res)
{
  return res.m_ritz_values.begin();
}

template <typename ResultsType>
static typename ResultsType::value_container::iterator endRitzValues(ResultsType &res)
{
  return res.m_ritz_values.end();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator beginRitzVectors(ResultsType &res)
{
  return res.m_ritz_vectors.begin();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator endRitzVectors(ResultsType &res)
{
  return res.m_ritz_vectors.end();
}

template <typename ValueType>
static void exposeArpack(const std::string &python_valuetypename, ValueType)
{
  using namespace boost::python;

  typedef typename arpack::results<ValueType> results_type;

  class_<results_type>
    (("ArpackResults"+python_valuetypename).c_str())
    .add_property("RitzValues", 
        range(beginRitzValues<results_type>, endRitzValues<results_type>))
    .add_property("RitzVectors", 
        range(beginRitzVectors<results_type>, endRitzVectors<results_type>))
    ;

  def("runArpack", arpack::doReverseCommunication<ValueType>,
      return_value_policy<manage_new_object>());
}




// main -----------------------------------------------------------------------
BOOST_PYTHON_MODULE(algorithms_internal)
{
  exposeMatrixOperators("Float64", double());
  exposeMatrixOperators("Complex64", std::complex<double>());

  exposeForAllMatrices(ublas_matrix_operator_exposer());

  python::enum_<arpack::which_eigenvalues>("arpack_which_eigenvalues")
    .value("SMALLEST_MAGNITUDE", arpack::SMALLEST_MAGNITUDE)
    .value("LARGEST_MAGNITUDE", arpack::LARGEST_MAGNITUDE)
    .value("SMALLEST_REAL_PART", arpack::SMALLEST_REAL_PART)
    .value("LARGEST_REAL_PART", arpack::LARGEST_REAL_PART)
    .value("SMALLEST_IMAGINARY_PART", arpack::SMALLEST_IMAGINARY_PART)
    .value("LARGEST_IMAGINARY_PART", arpack::LARGEST_IMAGINARY_PART)
    .export_values();

  python::enum_<arpack::arpack_mode>("arpack_mode")
    .value("REGULAR_NON_GENERALIZED", arpack::REGULAR_NON_GENERALIZED)
    .value("REGULAR_GENERALIZED", arpack::REGULAR_GENERALIZED)
    .value("SHIFT_AND_INVERT_GENERALIZED", arpack::SHIFT_AND_INVERT_GENERALIZED)
    .export_values();

  exposeArpack("Float64", double());
  exposeArpack("Complex64", std::complex<double>());
}
