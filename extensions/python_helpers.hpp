//
// Copyright (c) 2004-2006
// Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//




#ifndef HEADER_SEEN_PYTHON_HELPERS_H
#define HEADER_SEEN_PYTHON_HELPERS_H




#include <boost/python.hpp>




using boost::python::args;




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



#define PYTHON_ERROR(TYPE, REASON) \
{ \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw python::error_already_set(); \
}




#endif
