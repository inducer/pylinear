#include <boost/python.hpp>
#include <cg.h>
#include <umfpack.h>
#include <arpack.h>
#include "meta.h"



// shape accessor -------------------------------------------------------------
template <typename MatrixType>
static python::object getShape(const MatrixType &m)
{ 
  return python::make_tuple(m.size1(), m.size2());
}




// wrappers -------------------------------------------------------------------
template <typename ValueType>
class matrix_operator_wrapper : public matrix_operator<ValueType>
{
  public:
    typedef 
      typename matrix_operator<ValueType>::vector_type
      vector_type;

    matrix_operator_wrapper(PyObject *self, const matrix_operator<ValueType> &)
    { 
      // straight no-op.
    }

    unsigned size1() const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }
    unsigned size2() const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }
};




template <typename ValueType>
class algorithm_matrix_operator_wrapper : public algorithm_matrix_operator<ValueType>
{
  public:
    typedef 
      typename matrix_operator<ValueType>::vector_type
      vector_type;

    algorithm_matrix_operator_wrapper(PyObject *self, const algorithm_matrix_operator<ValueType> &)
    { 
      // straight no-op.
    }

    unsigned size1() const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }
    unsigned size2() const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }
};




template <typename ValueType>
class iterative_solver_matrix_operator_wrapper : public iterative_solver_matrix_operator<ValueType>
{
  public:
    typedef 
      typename matrix_operator<ValueType>::vector_type
      vector_type;

    iterative_solver_matrix_operator_wrapper(PyObject *self, const iterative_solver_matrix_operator<ValueType> &)
    { 
      // straight no-op.
    }

    unsigned size1() const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }
    unsigned size2() const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }

    void apply(const vector_type &before, vector_type &after) const
    {
      throw std::runtime_error("you can't implement matrix_operator in Python yet. Sorry." );
    }
};




// ublas_matrix_operator ------------------------------------------------------
template <typename MatrixType>
ublas_matrix_operator<MatrixType> *makeMatrixOperator(const MatrixType &mat)
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
void exposeForAllSimpleTypes(const std::string &python_eltname, const Exposer &exposer, ValueType)
{
  exposer.expose("Matrix" + python_eltname, ublas::matrix<ValueType>());
  exposer.expose("SparseExecuteMatrix" + python_eltname, ublas::compressed_matrix<ValueType>());
  exposer.expose("SparseBuildMatrix" + python_eltname, ublas::coordinate_matrix<ValueType>());
}




template <typename Exposer>
void exposeForAllMatrices(const Exposer &exposer, double)
{
  exposeForAllSimpleTypes("Float64", exposer, double());

  exposer.expose("SparseSymmetricExecuteMatrixFloat64", managed_symmetric_adaptor<
      ublas::compressed_matrix<double> >());
  exposer.expose("SparseSymmetricBuildMatrixFloat64", managed_symmetric_adaptor<
      ublas::coordinate_matrix<double> >());
}




template <typename Exposer>
void exposeForAllMatrices(const Exposer &exposer, std::complex<double>)
{
  exposeForAllSimpleTypes("Complex64", exposer, std::complex<double>());

  exposer.expose("SparseHermitianExecuteMatrixComplex64", managed_hermitian_adaptor<
      ublas::compressed_matrix<std::complex<double> > >());
  exposer.expose("SparseHermitianBuildMatrixComplex64", managed_hermitian_adaptor<
      ublas::coordinate_matrix<std::complex<double> > >());
}




template <typename Exposer>
void exposeForAllMatrices(const Exposer &exposer)
{
  exposeForAllMatrices(exposer, double());
  exposeForAllMatrices(exposer, std::complex<double>());
}




// main -----------------------------------------------------------------------
template <typename ValueType>
void exposeMatrixOperators(const std::string &python_eltname, ValueType)
{
  {
    typedef matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, matrix_operator_wrapper<ValueType> >
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
    algorithm_matrix_operator_wrapper<ValueType> >
      (("AlgorithmMatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("debug_level", &wrapped_type::getDebugLevel, &wrapped_type::setDebugLevel)
      .add_property("last_iteration_count", &wrapped_type::getLastIterationCount)
      ;
  }

  {
    typedef iterative_solver_matrix_operator<ValueType> wrapped_type;
    python::class_<wrapped_type, 
    python::bases<algorithm_matrix_operator<ValueType> >,
    iterative_solver_matrix_operator_wrapper<ValueType> >
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
       python::init<const ublas::identity_matrix<ValueType> &>());
    
    exposeForAllMatrices(
        umfpack_matrix_operator_constructor_exposer<wrapper_type>(pyclass), 
        ValueType());
  }
}




BOOST_PYTHON_MODULE(algorithms_internal)
{
  exposeMatrixOperators("Float64", double());
  exposeMatrixOperators("Complex64", std::complex<double>());

  exposeForAllMatrices(ublas_matrix_operator_exposer());
}
