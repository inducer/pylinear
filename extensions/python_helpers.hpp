#ifndef HEADER_SEEN_PYTHON_HELPERS_H
#define HEADER_SEEN_PYTHON_HELPERS_H




#include <boost/python.hpp>




template <typename T>
inline boost::python::handle<PyObject> handle_from_new_ptr(T *ptr)
{
  typename boost::python::manage_new_object::apply<T *>::type out_converter;
  return boost::python::handle<PyObject>(out_converter(ptr));
}




template <typename T>
inline boost::python::handle<PyObject> handle_from_object(const T &val)
{
  return boost::python::handle<PyObject>(boost::python::borrowed(boost::python::object(val).ptr()));
}




#endif
