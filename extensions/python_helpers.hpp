#ifndef HEADER_SEEN_PYTHON_HELPERS_H
#define HEADER_SEEN_PYTHON_HELPERS_H




#include <boost/python.hpp>




template <typename T>
inline PyObject *pyobject_from_new_ptr(T *ptr)
{
  return typename boost::python::manage_new_object::apply<T *>::type()(ptr);
}




template <typename T>
inline PyObject *pyobject_from_rvalue(const T &val)
{
  boost::python::object obj(val);
  PyObject *result = obj.ptr();
  Py_INCREF(result);
  return result;
}




#endif
