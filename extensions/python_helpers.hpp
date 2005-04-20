#ifndef HEADER_SEEN_PYTHON_HELPERS_H
#define HEADER_SEEN_PYTHON_HELPERS_H




#include <boost/python.hpp>




template <typename T>
inline PyObject *pyobject_from_new_ptr(T *ptr)
{
  return typename boost::python::manage_new_object::apply<T *>::type()(ptr);
}




inline PyObject *pyobject_from_object(const python::object &obj)
{
  PyObject *result = obj.ptr();
  Py_INCREF(result);
  return result;
}




template <typename T>
inline PyObject *pyobject_from_rvalue(const T &val)
{
  boost::python::object obj(val);
  return pyobject_from_object(obj);
}




#endif
