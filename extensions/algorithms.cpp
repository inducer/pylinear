#include <boost/python.hpp>
#include <cg.h>
#include <lu.h>
#include <cholesky.h>
#include <umfpack.h>
#include <arpack.h>
#include "meta.h"
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
/*
#include <boost/numeric/bindings/atlas/clapack.hpp>
#include <boost/numeric/ublas/lu.hpp>
*/




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




// cholesky_exposer -----------------------------------------------------------
struct cholesky_exposer
{
public:
  template <typename MatrixType>
  void expose(const std::string &python_mattype, MatrixType) const
  {
    python::def("cholesky", cholesky::cholesky<MatrixType, 
        typename strip_symmetric_wrappers<MatrixType>::type>,
        python::return_value_policy<python::manage_new_object>());
  }
};





// lu -------------------------------------------------------------------------
/*
 UBLAS builtin lu is dog-slow.
template <typename MatrixType>
python::object luWrapper(const MatrixType &a)
{
  using namespace ublas;

  typedef 
    typename strip_symmetric_wrappers<MatrixType>::type
    result_type;
  typedef
    permutation_matrix<unsigned>
    permut_type;

  std::auto_ptr<MatrixType> a_copy(new MatrixType(a));
  std::auto_ptr<permut_type> permut_ptr(new permut_type(a.size1()));

  axpy_lu_factorize(*a_copy, *permut_ptr);

  std::auto_ptr<result_type> l(new result_type(
        triangular_adaptor<MatrixType, unit_lower>(*a_copy)));
  std::auto_ptr<result_type> u(new result_type(
      triangular_adaptor<MatrixType, upper>(*a_copy)));

  int sign = 1;
  python::list py_permut;
  for (unsigned i = 0; i < permut_ptr->size(); i++)
  {
    // FIXME: BUG...
    py_permut.append((*permut_ptr)[i]);
    // FIXME: prove that this is right.
    if ((*permut_ptr)[i] != i) 
      sign *= -1;
  }
  
  python::object py_result = python::make_tuple(l.get(), u.get(), py_permut, sign);
  l.release();
  u.release();

  return py_result;
}
*/




/*
My LU is still slow, but faster that UBLAS builtin
*/
template <typename MatrixType>
python::object luWrapper(const MatrixType &a)
{
  typedef 
    typename strip_symmetric_wrappers<MatrixType>::type
    result_type;
  boost::tuple<result_type *, result_type *, std::vector<unsigned> *, int> result = 
    lu::lu<MatrixType, result_type, result_type>(a);
  std::auto_ptr<result_type> l(result.get<0>()), u(result.get<1>());
  std::auto_ptr<std::vector<unsigned> > permut_ptr(result.get<2>());
  std::vector<unsigned> &permut = *permut_ptr;

  python::list py_permut;
  for (unsigned i = 0; i < permut.size(); i++)
    py_permut.append(permut[i]);
  
  python::object py_result = python::make_tuple(l.get(), u.get(), py_permut, result.get<3>());
  l.release();
  u.release();

  return py_result;
}




/*
 couldn't find atlas CLAPACK 
template <typename MatrixType>
python::object luWrapper(const MatrixType &a)
{
  using namespace boost::numeric::bindings::atlas;

  typedef 
    typename strip_symmetric_wrappers<MatrixType>::type
    result_type;
  typedef
    ublas::vector<int>
    permut_type;

  std::auto_ptr<MatrixType> a_copy(new MatrixType(a));
  std::auto_ptr<permut_type> permut_ptr(new permut_type(a.size1()));

  lu_factor(*a_copy, *permut_ptr);

  std::auto_ptr<result_type> l(new result_type(
        ublas::triangular_adaptor<MatrixType, ublas::unit_lower>(*a_copy)));
  std::auto_ptr<result_type> u(new result_type(
        ublas::triangular_adaptor<MatrixType, ublas::upper>(*a_copy)));

  int sign = 1;
  python::list py_permut;
  for (unsigned i = 0; i < permut_ptr->size(); i++)
  {
    // FIXME: BUG...
    py_permut.append((*permut_ptr)[i]);
    // FIXME: prove that this is right.
    if ((*permut_ptr)[i] != i) 
      sign *= -1;
  }
  
  python::object py_result = python::make_tuple(l.get(), u.get(), py_permut, sign);
  l.release();
  u.release();

  return py_result;
}
*/




template <typename ValueType>
void exposeLU(ValueType)
{
  python::def("lu", luWrapper<ublas::matrix<ValueType> >);
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

  { 
    typedef umfpack::umfpack_matrix_operator<ValueType> wrapped_type;
    typedef 
      python::class_<wrapped_type, 
      python::bases<algorithm_matrix_operator<ValueType> >, boost::noncopyable>    
        wrapper_type;

    wrapper_type pyclass(("UMFPACKMatrixOperator"+python_eltname).c_str(), 
       python::init<const typename wrapped_type::matrix_type &>()
       [python::with_custodian_and_ward<1,2>()]);
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

  exposeLU(double());
  exposeLU(std::complex<double>());

  exposeForAllMatrices(cholesky_exposer());
}
