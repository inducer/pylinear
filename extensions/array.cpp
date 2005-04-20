#include <boost/python.hpp>
#include <boost/numeric/ublas/exception.hpp>
#include "meta.hpp"




namespace ublas = boost::numeric::ublas;




void pylinear_expose_sparse_build();
void pylinear_expose_sparse_ex();
void pylinear_expose_dense();
void pylinear_expose_vector();





namespace 
{
  void translate_divide_by_zero(const ublas::divide_by_zero &x)
  {
    PyErr_SetString(PyExc_ZeroDivisionError, x.what());
  }

  void translate_internal_logic(const ublas::internal_logic &x)
  {
    PyErr_SetString(PyExc_RuntimeError, x.what());
  }

  void translate_external_logic(const ublas::external_logic &x)
  {
    PyErr_SetString(PyExc_RuntimeError, x.what());
  }

  void translate_bad_size(const ublas::bad_size &x)
  {
    PyErr_SetString(PyExc_IndexError, x.what());
  }

  void translate_bad_index(const ublas::bad_index &x)
  {
    PyErr_SetString(PyExc_IndexError, x.what());
  }

  void translate_singular(const ublas::singular &x)
  {
    PyErr_SetString(PyExc_ValueError, x.what());
  }

  void translate_non_real(const ublas::non_real &x)
  {
    PyErr_SetString(PyExc_ValueError, x.what());
  }
}




BOOST_PYTHON_MODULE(_array)
{
  pylinear_expose_sparse_build();
  pylinear_expose_sparse_ex();
  pylinear_expose_dense();
  pylinear_expose_vector();

  python::register_exception_translator<ublas::divide_by_zero>(translate_divide_by_zero);
  python::register_exception_translator<ublas::internal_logic>(translate_internal_logic);
  python::register_exception_translator<ublas::external_logic>(translate_external_logic);
  python::register_exception_translator<ublas::bad_size>(translate_bad_size);
  python::register_exception_translator<ublas::bad_index>(translate_bad_index);
  python::register_exception_translator<ublas::singular>(translate_singular);
  python::register_exception_translator<ublas::non_real>(translate_non_real);
}
