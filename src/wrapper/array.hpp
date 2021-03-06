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




#include <complex>
#include <string>
#include <cmath>
#include <functional>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include "meta.hpp"
#include "python_helpers.hpp"

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/triangular.hpp>

#include <helpers.hpp>



using boost::python::class_;
using boost::python::handle;
using boost::python::borrowed;
using boost::python::enum_;
using boost::python::self;
using boost::python::def;
using helpers::decomplexify;




namespace {
// op wrappers ----------------------------------------------------------------
template <class Array, class Operator>
Array wrapUnaryOp(const Array &op)
{
  return Operator()(op);
}




template <class Array, class Operator>
Array wrapBinaryOp(const Array &op1, const Array &op2)
{
  return Operator()(op1, op2);
}




// helpers --------------------------------------------------------------------
template <typename T>
generic_ublas::minilist<T> getMinilist(const python::object &tup)
{
  unsigned len = python::extract<T>(tup.attr("__len__")());

  generic_ublas::minilist<T> result;
  for (unsigned i = 0; i < len; ++i)
    result.push_back(python::extract<T>(tup[i]));
  return result;
}




template <typename T>
python::tuple getPythonShapeTuple(const generic_ublas::minilist<T> &ml)
{
  if (ml.size() == 1)
    return python::make_tuple(ml[0]);
  else
    return python::make_tuple(ml[0], ml[1]);
}




template <typename T>
python::object getPythonIndexTuple(const generic_ublas::minilist<T> &ml)
{
  if (ml.size() == 1)
    return python::object(ml[0]);
  else
    return python::make_tuple(ml[0], ml[1]);
}




// shape accessors ------------------------------------------------------------
template <typename MatrixType>
inline unsigned getLength(const MatrixType &m)
{ 
  return m.size1();
}




template <typename ValueType>
inline unsigned getLength(const ublas::vector<ValueType> &m)
{ 
  return m.size();
}




template <typename MatrixType>
inline python::object getShape(const MatrixType &m)
{ 
  return getPythonShapeTuple(generic_ublas::getShape(m));
}




template <typename MatrixType>
inline void setShape(MatrixType &m, const python::tuple &new_shape)
{ 
  generic_ublas::setShape(m,getMinilist<typename MatrixType::size_type>(new_shape));
}




// iterator interface ---------------------------------------------------------
template <typename MatrixType>
struct python_matrix_key_iterator
{
  typename generic_ublas::matrix_iterator<MatrixType> m_iterator, m_end;

  python_matrix_key_iterator *iter()
  {
    return this;
  }

  python::object next()
  {
    if (m_iterator == m_end)
    {
      PyErr_SetNone(PyExc_StopIteration);
      throw python::error_already_set();
    }

    python::object result = getPythonIndexTuple(m_iterator.index());
    ++m_iterator;
    return result;
  }

  static python_matrix_key_iterator *obtain(MatrixType &m)
  {
    std::auto_ptr<python_matrix_key_iterator> it(new python_matrix_key_iterator);
    it->m_iterator = generic_ublas::begin(m);
    it->m_end = generic_ublas::end(m);
    return it.release();
  }
};




template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type >
struct python_matrix_value_iterator
{
  const MatrixType                      &m_matrix;
  typename MatrixType::size_type        m_row_index;

  python_matrix_value_iterator(const MatrixType &matrix)
  : m_matrix(matrix), m_row_index(0)
  {
  }

  python_matrix_value_iterator *iter()
  {
    return this;
  }

  handle<> next()
  {
    if (m_row_index >= m_matrix.size1())
    {
      PyErr_SetNone(PyExc_StopIteration);
      throw python::error_already_set();
    }

    return handle_from_new_ptr(
        new typename get_corresponding_vector_type<MatrixType>::type(
          ublas::row(m_matrix, m_row_index++)));
  }

  static python_matrix_value_iterator *obtain(MatrixType &m)
  {
    return new python_matrix_value_iterator(m);
  }
};




template <typename MatrixType> 
struct python_matrix_value_iterator<MatrixType, mpl::true_>
{
  const MatrixType                      &m_matrix;
  typename MatrixType::size_type        m_row_index;

  python_matrix_value_iterator(const MatrixType &matrix)
  : m_matrix(matrix), m_row_index(0)
  {
  }

  python_matrix_value_iterator *iter()
  {
    return this;
  }

  typename MatrixType::value_type next()
  {
    if (m_row_index >= m_matrix.size())
    {
      PyErr_SetNone(PyExc_StopIteration);
      throw python::error_already_set();
    }

    return m_matrix(m_row_index++);
  }

  static python_matrix_value_iterator *obtain(MatrixType &m)
  {
    return new python_matrix_value_iterator(m);
  }
};




// element accessors ----------------------------------------------------------
struct slice_info
{
  bool m_was_slice;
  Py_ssize_t m_start;
  Py_ssize_t m_end;
  Py_ssize_t m_stride;
  Py_ssize_t m_length;
};




void translateIndex(PyObject *slice_or_constant, slice_info &si, int my_length)
{
  si.m_was_slice = PySlice_Check(slice_or_constant);
  if (si.m_was_slice)
  {
    if (PySlice_GetIndicesEx(reinterpret_cast<PySliceObject *>(slice_or_constant), 
          my_length, &si.m_start, &si.m_end, &si.m_stride, &si.m_length) != 0)
      throw python::error_already_set();
  }
  else if (PyInt_Check(slice_or_constant))
  {
    int index = PyInt_AS_LONG(slice_or_constant);
    if (index < 0)
      index += my_length;
    if (index < 0)
      throw std::out_of_range("negative index out of bounds");
    if (index >= my_length)
      throw std::out_of_range("index out of bounds");
    si.m_start = index;
    si.m_end = index + 1;
    si.m_stride = 1;
    si.m_length = 1;
  }
  else
    throw std::out_of_range("invalid index object");
}




template <typename MatrixType>
handle<> getElement(/*const*/ MatrixType &m, handle<> index)
{ 
  typedef
    typename get_corresponding_vector_type<MatrixType>::type
    vector_t;
  typedef
    typename MatrixType::value_type
    value_t;
  typedef 
    ublas::basic_slice<typename MatrixType::size_type> slice_t;

  
  if (PyTuple_Check(index.get()))
  {
    // we have a tuple
    if (PyTuple_GET_SIZE(index.get()) != 2)
      PYTHON_ERROR(IndexError, "expected tuple of size 2");

    slice_info si1, si2;
    translateIndex(PyTuple_GET_ITEM(index.get(), 0), si1, m.size1());
    translateIndex(PyTuple_GET_ITEM(index.get(), 1), si2, m.size2());

    if (!si1.m_was_slice && !si2.m_was_slice)
      return handle_from_rvalue(value_t(m(si1.m_start, si2.m_start)));
    else if (!si1.m_was_slice)
      return handle_from_new_ptr(new vector_t(
            ublas::matrix_vector_slice<MatrixType>(m, 
              slice_t(si1.m_start, 0,            si2.m_length),
              slice_t(si2.m_start, si2.m_stride, si2.m_length))));
    else if (!si2.m_was_slice)
      return handle_from_new_ptr(new vector_t(
                  ublas::matrix_vector_slice<MatrixType>(m, 
                      slice_t(si1.m_start, si1.m_stride, si1.m_length),
                      slice_t(si2.m_start, 0,            si1.m_length))));
    else
    {
      return handle_from_new_ptr(
          new MatrixType(
            subslice(m,
                si1.m_start, si1.m_stride, si1.m_length,
                si2.m_start, si2.m_stride, si2.m_length)));
    }
  }
  else
  {
    slice_info si;
    translateIndex(index.get(), si, m.size1());

    if (!si.m_was_slice)
      return handle_from_new_ptr(new vector_t(row(m, si.m_start)));
    else
      return handle_from_new_ptr(
          new MatrixType(
            subslice(m,
              si.m_start, si.m_stride, si.m_length,
              0, 1, m.size2())
              ));
  }
}




template <typename ValueType>
handle<> getElement(/*const*/ ublas::vector<ValueType> &m, handle<> index)
{ 
  slice_info si;
  translateIndex(index.get(), si, m.size());

  if (!si.m_was_slice)
    return handle_from_rvalue(m(si.m_start));
  else
    return handle_from_new_ptr(
      new ublas::vector<ValueType>(subslice(m, si.m_start, si.m_stride, si.m_length)));
}




template <typename MatrixType>
void setElement(MatrixType &m, handle<> index, python::object &new_value)
{ 
  typedef 
    typename get_corresponding_vector_type<MatrixType>::type
    vector_t;
  typedef 
      typename MatrixType::value_type m_value_t;
  typedef 
      typename MatrixType::size_type m_size_t;
  typedef 
      ublas::basic_slice<m_size_t> slice_t;


  python::extract<typename MatrixType::value_type> new_scalar(new_value);
  python::extract<const vector_t &> new_vector(new_value);
  python::extract<const MatrixType &> new_matrix(new_value);

  if (PyTuple_Check(index.get()))
  {
    // we have a tuple
    if (PyTuple_GET_SIZE(index.get()) != 2)
      PYTHON_ERROR(IndexError, "expected tuple of size 2");

    slice_info si1, si2;
    translateIndex(PyTuple_GET_ITEM(index.get(), 0), si1, m.size1());
    translateIndex(PyTuple_GET_ITEM(index.get(), 1), si2, m.size2());

    if (new_scalar.check())
    {
      // scalar broadcast 
      subslice(m,
            si1.m_start, si1.m_stride, si1.m_length,
            si2.m_start, si2.m_stride, si2.m_length) =
        ublas::scalar_matrix<m_value_t>(si1.m_length, si2.m_length, new_scalar());
    }
    else if (new_vector.check())
    {
      const vector_t &new_vec(new_vector());
      if (si1.m_length == 1)
      {
	// replace row
        if (new_vec.size() != si2.m_length)
          PYTHON_ERROR(ValueError, "submatrix is wrong size for assignment");

        ublas::matrix_vector_slice<MatrixType>(m,
            slice_t(si1.m_start, 0,            si2.m_length),
            slice_t(si2.m_start, si2.m_stride, si2.m_length)) = new_vec;
      }
      else if (si2.m_length == 1)
      {
	// replace column
        if (new_vector().size() != si1.m_length)
          PYTHON_ERROR(ValueError, "submatrix is wrong size for assignment");

        ublas::matrix_vector_slice<MatrixType>(m,
            slice_t(si1.m_start, si1.m_stride, si1.m_length),
            slice_t(si2.m_start, 0,            si1.m_length)) = new_vec;
      }
      else
      {
        // broadcast vector across matrix
        ublas::matrix_slice<MatrixType> my_slice(m,
              slice_t(si1.m_start, si1.m_stride, si1.m_length),
              slice_t(si2.m_start, si2.m_stride, si2.m_length));

        if (new_vec.size() != my_slice.size2())
          PYTHON_ERROR(ValueError, "submatrix is wrong size for assignment");

        for (m_size_t i = 0; i < my_slice.size1(); ++i)
          row(my_slice, i) = new_vector();
      }
    }
    else if (new_matrix.check())
    {
      // no broadcast
      const MatrixType &new_mat = new_matrix();
      if (int(new_mat.size1()) != si1.m_length || 
          int(new_mat.size2()) != si2.m_length)
        PYTHON_ERROR(ValueError, "submatrix is wrong size for assignment");

      subslice(m,
        si1.m_start, si1.m_stride, si1.m_length,
        si2.m_start, si2.m_stride, si2.m_length) = new_mat;
    }
    else
      PYTHON_ERROR(ValueError, "unknown type in element or slice assignment");
  }
  else
  {
    slice_info si;
    translateIndex(index.get(), si, m.size1());

    if (new_scalar.check())
      subslice(m,
          si.m_start, si.m_stride, si.m_length,
          0, 1, m.size2()) = 
        ublas::scalar_matrix<m_value_t>(si.m_length, m.size2(), new_scalar());
    else if (new_vector.check())
    {
      vector_t new_vec = new_vector();

      if (si.m_length == 1)
      {
        if (new_vec.size() != m.size2())
          PYTHON_ERROR(ValueError, "submatrix is wrong size for assignment");

        row(m,si.m_start) = new_vec;
      }
      else
      {
        // broadcast vector across matrix
        if (new_vec.size() != m.size2())
          PYTHON_ERROR(ValueError, "submatrix is wrong size for assignment");

        for (m_size_t i = si.m_start; i < si.m_end; i += si.m_stride)
          row(m, i) = new_vec;
      }
    }
    else if (new_matrix.check())
    {
      const MatrixType &new_mat = new_matrix();

      if (int(new_mat.size1()) != si.m_length || 
          int(new_mat.size2()) != m.size2())
        PYTHON_ERROR(ValueError, "submatrix is wrong size for assignment");

      project(m,
          ublas::basic_slice<typename MatrixType::size_type>(si.m_start, si.m_stride, si.m_length),
          ublas::basic_slice<typename MatrixType::size_type>(0, 1, m.size2())) = new_mat;
    }
    else
      PYTHON_ERROR(ValueError, "unknown type in element or slice assignment");
  }
}




template <typename ValueType>
void setElement(ublas::vector<ValueType> &m, handle<> index, python::object &new_value)
{ 
  python::extract<ValueType> new_scalar(new_value);
  python::extract<const ublas::vector<ValueType> &> new_vector(new_value);

  slice_info si;
  translateIndex(index.get(), si, m.size());

  if (new_scalar.check())
    subslice(m, si.m_start, si.m_stride, si.m_length) =
      ublas::scalar_vector<ValueType>(si.m_length, new_scalar());
  else if (new_vector.check())
  {
    if (new_vector().size() != si.m_length)
      PYTHON_ERROR(ValueError, "subvector is wrong size for assignment");
    subslice(m, si.m_start, si.m_stride, si.m_length) = 
      new_vector();
  }
  else if (!PyInt_Check(index.get())) // only for slice indices
  {
    python::stl_input_iterator<ValueType> it(new_value);
    // try iterating over new_value, maybe it's a list
    for (unsigned i = 0; i < si.m_length; i++)
      m[si.m_start + si.m_stride*i] = *it++;
  }
  else
    PYTHON_ERROR(ValueError, "Unknown type in element or slice assignment");
}




// pickling -------------------------------------------------------------------
template <typename MatrixType>
struct sparse_pickle_suite : python::pickle_suite
{
  static
  python::tuple
  getinitargs(const MatrixType &m)
  {
    return getPythonShapeTuple(generic_ublas::getShape(m));
  }




  static
  python::object
  getstate(MatrixType &m)
  {
    generic_ublas::matrix_iterator<MatrixType>
      first = generic_ublas::begin(m),
      last = generic_ublas::end(m);
    
    python::list result;
    while (first != last)
    {
      result.append(python::make_tuple(getPythonIndexTuple(first.index()),
				       typename MatrixType::value_type(*first)));
      first++;
    }

    return result;
  }




  static 
  void
  setstate(MatrixType &m, python::object entries)
  {
    unsigned len = python::extract<unsigned>(entries.attr("__len__")());
    for (unsigned i = 0; i < len; i++)
    {
      generic_ublas::insert_element(
        m,
        getMinilist<typename MatrixType::size_type>(
          python::extract<python::tuple>(entries[i][0])),
        python::extract<typename MatrixType::value_type>(entries[i][1]));
    }
  }
};




template <typename MatrixType>
struct dense_pickle_suite : python::pickle_suite
{
  static
  python::tuple
  getinitargs(const MatrixType &m)
  {
    return getPythonShapeTuple(generic_ublas::getShape(m));
  }




  static 
  python::object
  getstate(MatrixType &m)
  {
    generic_ublas::matrix_iterator<MatrixType>
      first = generic_ublas::begin(m),
      last = generic_ublas::end(m);
    
    python::list result;
    while (first != last)
      result.append(*first++);
    
    return result;
  }
  



  static void
  setstate(MatrixType &m, python::object entries)
  {
    generic_ublas::matrix_iterator<MatrixType> 
      first = generic_ublas::begin(m);
    
    unsigned len = python::extract<unsigned>(entries.attr("__len__")());
    for (unsigned i = 0; i < len; i++)
      *first++ = (typename MatrixType::value_type)
	python::extract<typename MatrixType::value_type>(entries[i]);
  }
};
  



template <typename PythonClass, typename WrappedClass>
void exposePickling(PythonClass &pyclass, WrappedClass)
{
  pyclass.def_pickle(sparse_pickle_suite<WrappedClass>());
}




template <typename PythonClass, typename V>
void exposePickling(PythonClass &pyclass, ublas::matrix<V>)
{
  pyclass.def_pickle(dense_pickle_suite<ublas::matrix<V> >());
}




template <typename PythonClass, typename V>
void exposePickling(PythonClass &pyclass, ublas::vector<V>)
{
  pyclass.def_pickle(dense_pickle_suite<ublas::vector<V> >());
}




// specialty constructors -----------------------------------------------------
template <typename MatrixType>
MatrixType *getFilledMatrix(
    typename MatrixType::size_type size1, 
    typename MatrixType::size_type size2, 
    const typename MatrixType::value_type &value)
{
  typedef typename MatrixType::value_type value_t;

  std::auto_ptr<MatrixType> mat(new MatrixType(size1, size2));
  *mat = ublas::scalar_matrix<value_t>(size1, size2, value);
  return mat.release();
}




template <typename MatrixType>
MatrixType *getFilledVector(
    typename MatrixType::size_type size1, 
    const typename MatrixType::value_type &value)
{
  typedef typename MatrixType::value_type value_t;

  std::auto_ptr<MatrixType> mat(new MatrixType(size1));
  *mat = ublas::scalar_vector<value_t>(size1, value);
  return mat.release();
}




// universal functions --------------------------------------------------------
template <typename MatrixType>
inline MatrixType *copyNew(const MatrixType &m)
{
  return new MatrixType(m);
}




template <typename MatrixType>
handle<> hermiteMatrix(const MatrixType &m)
{
  return handle_from_new_ptr(new MatrixType(herm(m)));
}




template <typename MatrixType>
handle<> transposeMatrix(const MatrixType &m)
{
  return handle_from_new_ptr(new MatrixType(trans(m)));
}




template <typename VectorType>
handle<> hermiteVector(const VectorType &m)
{
  return handle_from_new_ptr(new VectorType(conj(m)));
}




template <typename VectorType>
handle<> transposeVector(const VectorType &m)
{
  return handle_from_new_ptr(new VectorType(m));
}




template <typename MatrixType>
struct realWrapper
{
  typedef 
    typename change_value_type<MatrixType, 
      typename decomplexify<typename MatrixType::value_type>::type>::type
    result_type;

  inline static handle<> apply(const MatrixType &m)
  {
    return handle_from_new_ptr(new result_type(real(m)));
  }
};




template <typename MatrixType>
struct imagWrapper
{
  typedef 
    typename change_value_type<MatrixType, 
      typename decomplexify<typename MatrixType::value_type>::type>::type
    result_type;

  inline static handle<> apply(const MatrixType &m)
  {
    return handle_from_new_ptr(new result_type(imag(m)));
  }
};




template <typename MatrixType>
struct conjugateWrapper
{
  typedef MatrixType result_type;

  inline static result_type *apply(const MatrixType &m)
  {
    return new result_type(conj(m));
  }
};




template <typename MatrixType>
inline
void add_element_inplace(MatrixType &mat, 
    typename MatrixType::size_type i,
    typename MatrixType::size_type j,
    typename MatrixType::value_type x)
{
  mat(i,j) += x;
}




template <typename V>
inline
void add_element_inplace(ublas::coordinate_matrix<V, ublas::column_major> &mat, 
    unsigned i,
    unsigned j,
    V x)
{
  mat.append_element(i, j, x);
}




template <typename MatrixType, typename SmallMatrixType>
void addBlock(MatrixType &mat, 
    typename MatrixType::size_type start_row,
    typename MatrixType::size_type start_column,
    SmallMatrixType &small_mat)
{
  typedef typename SmallMatrixType::size_type index_t;

  generic_ublas::matrix_iterator<SmallMatrixType>
    first = generic_ublas::begin(small_mat),
    last = generic_ublas::end(small_mat);

  while (first != last)
  {
    const generic_ublas::minilist<index_t> index = first.index();
    add_element_inplace(mat, 
        start_row+index[0], 
        start_column+index[1], 
        *first++);
  }
}




template <typename MatrixType, typename SmallMatrixType>
void addScattered(MatrixType &mat, 
    python::object row_indices_py, 
    python::object column_indices_py,
    SmallMatrixType &small_mat)
{
  using namespace boost::python;

  typedef typename SmallMatrixType::size_type index_t;
  std::vector<index_t> row_indices;
  std::vector<index_t> column_indices;
  copy(
      stl_input_iterator<index_t>(row_indices_py),
      stl_input_iterator<index_t>(),
      back_inserter(row_indices));
  copy(
      stl_input_iterator<index_t>(column_indices_py),
      stl_input_iterator<index_t>(),
      back_inserter(column_indices));

  if (row_indices.size() != small_mat.size1()
      || column_indices.size() != small_mat.size2())
    throw std::runtime_error("sizes don't match");

  generic_ublas::matrix_iterator<SmallMatrixType>
    first = generic_ublas::begin(small_mat),
    last = generic_ublas::end(small_mat);

  while (first != last)
  {
    const generic_ublas::minilist<index_t> index = first.index();
    add_element_inplace(mat, 
        row_indices[index[0]], 
        column_indices[index[1]], 
        *first++);
  }
}




template <typename MatrixType, typename SmallMatrixType>
void addScatteredWithSkip(MatrixType &mat, 
    python::object row_indices_py, 
    python::object column_indices_py,
    SmallMatrixType &small_mat)
{
  using namespace boost::python;

  typedef typename SmallMatrixType::size_type index_t;
  std::vector<index_t> row_indices;
  std::vector<index_t> column_indices;
  copy(
      stl_input_iterator<index_t>(row_indices_py),
      stl_input_iterator<index_t>(),
      back_inserter(row_indices));
  copy(
      stl_input_iterator<index_t>(column_indices_py),
      stl_input_iterator<index_t>(),
      back_inserter(column_indices));

  if (row_indices.size() != small_mat.size1()
      || column_indices.size() != small_mat.size2())
    throw std::runtime_error("sizes don't match");

  generic_ublas::matrix_iterator<SmallMatrixType>
    first = generic_ublas::begin(small_mat),
    last = generic_ublas::end(small_mat);

  while (first != last)
  {
    const generic_ublas::minilist<index_t> index = first.index();
    unsigned dest_row = row_indices[index[0]];
    unsigned dest_col = column_indices[index[1]];
    if (dest_row >= 0 && dest_col >= 0)
      add_element_inplace(mat, dest_row, dest_col, *first);
    ++first;
  }
}




template <typename MatrixType>
typename get_corresponding_vector_type<MatrixType>::type *
solveLower(const MatrixType &mat, 
    const typename get_corresponding_vector_type<MatrixType>::type &vec)
{
  return new 
    typename get_corresponding_vector_type<MatrixType>::type
    (ublas::solve(mat, vec, ublas::lower_tag()));
}




template <typename MatrixType>
typename get_corresponding_vector_type<MatrixType>::type *
solveUpper(const MatrixType &mat, 
    const typename get_corresponding_vector_type<MatrixType>::type &vec)
{
  return new 
    typename get_corresponding_vector_type<MatrixType>::type
    (ublas::solve(mat, vec, ublas::upper_tag()));
}




#ifndef PYLINEAR_NO_UFUNCS
namespace ufuncs
{
  // my_XX function prototypes ------------------------------------------------
  template <typename T>
  static inline T my_arg(T x)
  {
    if (x >= 0)
      return 0;
    else
      return M_PI;
  }

  template <typename T>
  static inline T my_arg(const std::complex<T> &x)
  {
    return std::arg(x);
  }

  template <typename T>
  static inline T my_abs(T x)
  {
    return fabs(x);
  }

  template <typename T>
  static inline T my_abs(const std::complex<T> &x)
  {
    return abs(x);
  }

  template <typename T>
  static inline T my_floor(T x)
  {
    return floor(x);
  }

  template <typename T>
  static inline std::complex<T> my_floor(const std::complex<T> &x)
  {
    return std::complex<T>(floor(x.real()), floor(x.imag()));
  }

  template <typename T>
  static inline T my_ceil(T x)
  {
    return ceil(x);
  }

  template <typename T>
  static inline std::complex<T> my_ceil(const std::complex<T> &x)
  {
    return std::complex<T>(ceil(x.real()), ceil(x.imag()));
  }

  template <typename T>
  static inline bool my_less(T x, T y)
  {
    return x < y;
  }

  template <typename T>
  static inline bool my_less(const std::complex<T> &x, const std::complex<T> &y)
  {
    return x.real() < y.real();
  }



  // binary ufuncs ------------------------------------------------------------
  template <typename T>
  struct power : public std::binary_function<T, T, T>
  {
    T operator()(T a, T b) { return pow(a, b); }
  };




  template <typename T>
  struct minimum : public std::binary_function<T, T, T>
  {
    T operator()(T a, T b) { if (my_less(a,b)) return a; else return b; }
  };




  template <typename T>
  struct maximum : public std::binary_function<T, T, T>
  {
    T operator()(T a, T b) { if (my_less(a,b)) return b; else return a; }
  };




  // identical types ----------------------------------------------------------
#define MAKE_UNARY_FUNCTION_ADAPTER(NAME, f) \
  template <typename T> \
  struct simple_function_adapter_##NAME \
  { \
    typedef T result_type; \
    \
    inline result_type operator()(const T &x) \
    { \
      return f(x); \
    } \
  }

  // every type
  MAKE_UNARY_FUNCTION_ADAPTER(cos, cos);
  MAKE_UNARY_FUNCTION_ADAPTER(cosh, cosh);
  MAKE_UNARY_FUNCTION_ADAPTER(exp, exp);
  MAKE_UNARY_FUNCTION_ADAPTER(log, log);
  MAKE_UNARY_FUNCTION_ADAPTER(log10, log10);
  MAKE_UNARY_FUNCTION_ADAPTER(sin, sin);
  MAKE_UNARY_FUNCTION_ADAPTER(sinh, sinh);
  MAKE_UNARY_FUNCTION_ADAPTER(sqrt, sqrt);
  MAKE_UNARY_FUNCTION_ADAPTER(tan, tan);
  MAKE_UNARY_FUNCTION_ADAPTER(tanh, tanh);

  MAKE_UNARY_FUNCTION_ADAPTER(floor, my_floor);
  MAKE_UNARY_FUNCTION_ADAPTER(ceil, my_ceil);
#undef MAKE_UNARY_FUNCTION_ADAPTER




  // to real ------------------------------------------------------------------
#define MAKE_UNARY_TO_REAL_FUNCTION_ADAPTER(NAME, f) \
  template <typename T> \
  struct simple_function_adapter_##NAME \
  { \
    typedef typename decomplexify<T>::type result_type; \
    \
    inline result_type operator()(const T &x) \
    { \
      return f(x); \
    } \
  }

  MAKE_UNARY_TO_REAL_FUNCTION_ADAPTER(absolute, my_abs);
  MAKE_UNARY_TO_REAL_FUNCTION_ADAPTER(arg, my_arg);
#undef MAKE_UNARY_TO_REAL_FUNCTION_ADAPTER



  template <typename MatrixType, typename Function>
  struct unary_ufunc_applicator
  {
    typedef 
      typename change_value_type<MatrixType, typename Function::result_type>::type
      result_type;

    static result_type *apply(MatrixType &m)
    {
      Function f;

      std::auto_ptr<result_type> new_mat(
          generic_ublas::newWithShape<result_type>(generic_ublas::getShape(m)));

      // FIXME: should be const
      generic_ublas::matrix_iterator<MatrixType>
        first = generic_ublas::begin(m), last = generic_ublas::end(m);

      while (first != last)
      {
        generic_ublas::set(*new_mat, first.index(), f(*first));
        ++first;
      }

      return new_mat.release();
    }
  };




  // binary ufuncs ------------------------------------------------------------
  template <class Function>
  struct reverse_binary_function : 
  public std::binary_function<typename Function::second_argument_type,
  typename Function::first_argument_type, typename Function::result_type>
  {
    typename Function::result_type
    operator()(
        const typename Function::second_argument_type &a2, 
        const typename Function::first_argument_type &a1)
    {
      return Function()(a1, a2);
    }
  };




  // neutral element detection
  // why on earth do this? -- allowing +0 without penalty removes an
  // abstraction penalty in Python

  template <typename Func>
  struct neutral_element
  {
    static const bool has_neutral_second_argument = false;
    static typename Func::second_argument_type get()
    {
      throw std::runtime_error("tried to get non-existent neutral element");
    }
  };

  template <typename T>
  struct neutral_element<std::plus<T> >
  {
    static const bool has_neutral_second_argument = true;
    static T get()
    { return 0; }
  };

  template <typename T>
  struct neutral_element<std::minus<T> >
  {
    static const bool has_neutral_second_argument = true;
    static T get()
    { return 0; }
  };

  template <typename T>
  struct neutral_element<std::multiplies<T> >
  {
    static const bool has_neutral_second_argument = true;
    static T get()
    { return 1; }
  };

  template <typename T>
  struct neutral_element<std::divides<T> >
  {
    static const bool has_neutral_second_argument = true;
    static T get()
    { return 1; }
  };
  
  template <typename Function, typename MatrixType>
  handle<>
  applyBackend(Function f, MatrixType &m1, python::object obj)
  {
    typedef 
      typename MatrixType::value_type
      value_type;
    typedef 
      ublas::vector<value_type>
      vector_type;
    typedef 
      typename decomplexify<value_type>::type
      nc_value_type;
    typedef 
      typename MatrixType::const_iterator1 
      it1_t;
    typedef 
      typename MatrixType::const_iterator2 
      it2_t;

    python::extract<const MatrixType &> m2_extractor(obj);
    python::extract<const vector_type &> v2_extractor(obj);
    python::extract<value_type> s2_extractor(obj);
    python::extract<nc_value_type> n2_extractor(obj);

    std::auto_ptr<MatrixType> new_mat(new MatrixType(m1.size1(), m1.size2()));

    if (m2_extractor.check())
    {
      const MatrixType &m2 = m2_extractor();

      if (m1.size1() != m2.size1() || m1.size2() != m2.size2())
        throw std::runtime_error("cannot apply binary ufunc to arrays of different sizes");

      for (it1_t it1 = m1.begin1(); it1 != m1.end1(); ++it1) 
        for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
          new_mat->insert_element(it2.index1(), it2.index2(), 
                          f(*it2, m2(it2.index1(), it2.index2())));
    }
    else if (v2_extractor.check())
    {
      const vector_type &v2 = v2_extractor();

      if (m1.size2() != v2.size())
        throw std::runtime_error("cannot apply binary ufunc to arrays of different sizes");

      for (it1_t it1 = m1.begin1(); it1 != m1.end1(); ++it1) 
        for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
          new_mat->insert_element(it2.index1(), it2.index2(), 
                          f(*it2, v2(it2.index2())));
    }
    else if (s2_extractor.check())
    {
      value_type s2 = s2_extractor();
      if (neutral_element<Function>::has_neutral_second_argument
              && s2 == neutral_element<Function>::get())
        return handle_from_existing_ref(m1);

      for (it1_t it1 = m1.begin1(); it1 != m1.end1(); ++it1) 
        for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
          new_mat->insert_element(it2.index1(), it2.index2(), f(*it2, s2));
    }
    else if (n2_extractor.check())
    {
      value_type n2 = n2_extractor();

      for (it1_t it1 = m1.begin1(); it1 != m1.end1(); ++it1) 
        for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
          new_mat->insert_element(it2.index1(), it2.index2(), f(*it2, n2));
    }
    else
      return handle<>(borrowed(Py_NotImplemented));

    return handle_from_new_ptr(new_mat.release());
  }

  template <typename RealFunction, typename V>
  handle<>
  applyBackend(RealFunction f, ublas::vector<V> &m1, python::object obj)
  {
    typedef 
      typename ublas::vector<V>
      vector_type;
    typedef 
      typename vector_type::value_type
      value_type;
    typedef 
      typename decomplexify<value_type>::type
      nc_value_type;
    typedef 
      typename vector_type::const_iterator
      it_t;

    python::extract<const vector_type &> m2_extractor(obj);
    python::extract<value_type> s2_extractor(obj);
    python::extract<nc_value_type> n2_extractor(obj);

    std::auto_ptr<vector_type> new_vec(new vector_type(m1.size()));

    if (m2_extractor.check())
    {
      const vector_type &m2 = m2_extractor();

      if (m1.size() != m2.size())
        throw std::runtime_error("cannot apply binary ufunc to vectors of different sizes");

      for (it_t it = m1.begin(); it != m1.end(); ++it) 
        new_vec->insert_element(it.index(), f(*it, m2(it.index())));
    }
    else if (s2_extractor.check())
    {
      value_type s2 = s2_extractor();

      if (neutral_element<RealFunction>::has_neutral_second_argument
              && s2 == neutral_element<RealFunction>::get())
        return handle_from_existing_ref(m1);

      for (it_t it = m1.begin(); it != m1.end(); ++it) 
        new_vec->insert_element(it.index(), f(*it, s2));
    }
    else if (n2_extractor.check())
    {
      value_type n2 = n2_extractor();

      for (it_t it = m1.begin(); it != m1.end(); ++it) 
        new_vec->insert_element(it.index(), f(*it, n2));
    }
    else
      return handle<>(borrowed(Py_NotImplemented));

    return handle_from_new_ptr(new_vec.release());
  }

  template<typename Function, typename MatrixType, typename Name>
  handle<> apply(python::object op1, python::object op2)
  {
    handle<> result = applyBackend(
        Function(), 
        python::extract<MatrixType&>(op1)(), 
        op2);
    if (result.get() != Py_NotImplemented)
      return result;
    else
      return handle<>(PyObject_CallMethod(op1.ptr(), 
          (char *) "_cast_and_retry", 
          (char *) "sO",
          (std::string("_ufunc_") + Name::m_name).c_str(), 
          op2.ptr()));
  }

  template<typename Function, typename MatrixType>
  handle<> applyWithoutCoercion(MatrixType &op1, python::object op2)
  {
    return applyBackend(Function(), op1, op2);
  }

  template<typename Function, typename MatrixType, typename Name>
  handle<> applyReversed(python::object op1, python::object op2)
  {
    handle<> result = applyBackend(
        reverse_binary_function<Function>(), 
        python::extract<MatrixType &>(op1)(), 
        op2);
    if (result.get() != Py_NotImplemented)
      return result;
    else
      return handle<>(PyObject_CallMethod(op1.ptr(), 
          (char *) "_cast_and_retry", 
          (char *) "sO",
          (std::string("_reverse_ufunc_") + Name::m_name).c_str(), 
          op2.ptr()));
  }

  template<typename Function, typename MatrixType>
  handle<> applyReversedWithoutCoercion(MatrixType &op1, python::object op2)
  {
    return applyBackend(
        reverse_binary_function<Function>(), 
        op1, 
        op2);
  }

  #define DECLARE_NAME_STRUCT(NAME) \
  struct name_##NAME { static const char *m_name; }; const char *name_##NAME::m_name = #NAME;

  DECLARE_NAME_STRUCT(add);
  DECLARE_NAME_STRUCT(subtract);
  DECLARE_NAME_STRUCT(multiply);
  DECLARE_NAME_STRUCT(divide);
  DECLARE_NAME_STRUCT(divide_safe);
  DECLARE_NAME_STRUCT(power);
  DECLARE_NAME_STRUCT(maximum);
  DECLARE_NAME_STRUCT(minimum);
}
#endif // PYLINEAR_NO_UFUNCS








// wrapper for stuff that is common to vectors and matrices -------------------
template <typename MatrixType>
typename MatrixType::value_type
sum(MatrixType &mat)
{
  generic_ublas::matrix_iterator<MatrixType>
    first = generic_ublas::begin(mat),
    last = generic_ublas::end(mat);
    
  typename MatrixType::value_type result = 0;
  while (first != last)
    result += *first++;
  return result;
}




template <typename MatrixType>
typename helpers::decomplexify<typename MatrixType::value_type>::type
abs_square_sum(MatrixType &mat)
{
  generic_ublas::matrix_iterator<MatrixType>
    first = generic_ublas::begin(mat),
    last = generic_ublas::end(mat);
    
  typedef 
    typename helpers::decomplexify<typename MatrixType::value_type>::type 
    real_type;
  real_type result = 0;
  while (first != last)
    result += helpers::absolute_value_squared(*first++);
  return result;
}




template <typename MatrixType>
typename MatrixType::value_type
product(MatrixType &mat)
{
  generic_ublas::matrix_iterator<MatrixType>
    first = generic_ublas::begin(mat),
    last = generic_ublas::end(mat);
    
  typename MatrixType::value_type result = 1;
  while (first != last)
    result *= *first++;
  return result;
}




template <typename WrappedClass, typename PythonClass>
void exposeUfuncs(PythonClass &pyclass)
{
#ifndef PYLINEAR_NO_UFUNCS
  typedef
    typename WrappedClass::value_type
    value_type;

  pyclass
    .def("_ufunc_conjugate", conjugateWrapper<WrappedClass>::apply,
	python::return_value_policy<python::manage_new_object>())
    .add_property("real", realWrapper<WrappedClass>::apply,
        "Return real part of the Array.")
    .add_property("imaginary", imagWrapper<WrappedClass>::apply,
        "Return imaginary part of the Array.");

#define MAKE_UNARY_UFUNC(f) \
  pyclass.def("_ufunc_" #f, ufuncs::unary_ufunc_applicator<WrappedClass, \
      ufuncs::simple_function_adapter_##f<value_type> >::apply, \
      python::return_value_policy<python::manage_new_object>());
  MAKE_UNARY_UFUNC(cos);
  MAKE_UNARY_UFUNC(cosh);
  MAKE_UNARY_UFUNC(exp);
  MAKE_UNARY_UFUNC(log);
  MAKE_UNARY_UFUNC(log10);
  MAKE_UNARY_UFUNC(sin);
  MAKE_UNARY_UFUNC(sinh);
  MAKE_UNARY_UFUNC(sqrt);
  MAKE_UNARY_UFUNC(tan);
  MAKE_UNARY_UFUNC(tanh);
  MAKE_UNARY_UFUNC(floor);
  MAKE_UNARY_UFUNC(ceil);

  MAKE_UNARY_UFUNC(arg);
  MAKE_UNARY_UFUNC(absolute);
#undef MAKE_UNARY_UFUNC

#define MAKE_BINARY_UFUNC(NAME, f) \
  pyclass.def("_ufunc_" #NAME, ufuncs::apply \
          <f<value_type>, WrappedClass, ufuncs::name_##NAME>); \
  pyclass.def("_nocast__ufunc_" #NAME, ufuncs::applyWithoutCoercion \
          <f<value_type>, WrappedClass>);
#define MAKE_REVERSE_BINARY_UFUNC(NAME, f) \
  pyclass.def("_reverse_ufunc_" #NAME, ufuncs::applyReversed \
          <f<value_type>, WrappedClass, ufuncs::name_##NAME>); \
  pyclass.def("_nocast__reverse_ufunc_" #NAME, ufuncs::applyReversedWithoutCoercion \
          <f<value_type>, WrappedClass>);
  MAKE_BINARY_UFUNC(add, std::plus);
  MAKE_BINARY_UFUNC(subtract, std::minus);
  MAKE_REVERSE_BINARY_UFUNC(subtract, std::minus);
  MAKE_BINARY_UFUNC(multiply, std::multiplies);
  MAKE_BINARY_UFUNC(divide, std::divides);
  MAKE_REVERSE_BINARY_UFUNC(divide, std::divides);
  MAKE_BINARY_UFUNC(divide_safe, std::divides); // FIXME: bogus
  MAKE_REVERSE_BINARY_UFUNC(divide_safe, std::divides); // FIXME: bogus
  MAKE_BINARY_UFUNC(power, ufuncs::power);
  MAKE_REVERSE_BINARY_UFUNC(power, ufuncs::power);
  MAKE_BINARY_UFUNC(maximum, ufuncs::maximum);
  MAKE_BINARY_UFUNC(minimum, ufuncs::minimum);
#undef MAKE_BINARY_UFUNC
#endif // PYLINEAR_NO_UFUNCS
}




template <typename PythonClass, typename WrappedClass>
void exposeElementWiseBehavior(PythonClass &pyclass, WrappedClass)
{
  typedef WrappedClass cl;
  typedef typename cl::value_type value_type;
  pyclass
    .def("copy", copyNew<cl>, 
        python::return_value_policy<python::manage_new_object>(),
        "Return an exact copy of the given Array.")
    .def("clear", &cl::clear,
        "Discard Array content and fill with zeros, if necessary.")

    .add_property(
      "shape", 
      (python::object (*)(const cl &)) getShape, 
      (void (*)(cl &, const python::tuple &)) setShape,
      "Return a shape tuple for the Array.")
    .add_property(
      "__array_shape__", 
      (python::object (*)(const cl &)) getShape)
    .def("__len__", (unsigned (*)(const cl &)) getLength,
        "Return the length of the leading dimension of the Array.")
    .def("swap", &cl::swap)

    .def("__getitem__", (handle<> (*)(/*const*/ cl &, handle<>)) getElement)
    .def("__setitem__", (void (*)(cl &, handle<>, python::object &)) setElement)
    ;

  // unary negation
  pyclass
    .def("__neg__", wrapUnaryOp<cl, std::negate<cl> >)
    ;

  // container - container
  pyclass
    .def(self += self)
    .def(self -= self)

    .def("sum", sum<cl>,
        "Return the sum of the Array's entries.")
    .def("_product_nonzeros", product<cl>,
        "Return the product of the Array's entries, excluding zeros in sparse Arrays.")
    .def("abs_square_sum", abs_square_sum<cl>)
    ;

  exposePickling(pyclass, WrappedClass());
}




template <typename PythonClass, typename WrappedClass>
void exposeIterator(PythonClass &pyclass, const std::string &python_typename, WrappedClass)
{
  typedef 
    python_matrix_value_iterator<WrappedClass>
    value_iterator;

  typedef 
    python_matrix_key_iterator<WrappedClass>
    key_iterator;

  pyclass
    .def("__iter__", &value_iterator::obtain,
        python::return_value_policy<python::manage_new_object,
        python::return_internal_reference<> >())
    .def("indices", &key_iterator::obtain,
        python::return_value_policy<python::manage_new_object,
        python::return_internal_reference<> >(),
        "Return an iterator over all non-zero index pairs of the Array.")
    ;

  class_<key_iterator>
    ((python_typename + "KeyIterator").c_str(), python::no_init)
    .def("next", &key_iterator::next)
    .def("__iter__", &key_iterator::iter,
        python::return_self<>())
    ;

  class_<value_iterator>
    ((python_typename + "ValueIterator").c_str(), python::no_init)
    .def("next", &value_iterator::next)
    .def("__iter__", &value_iterator::iter,
        python::return_self<>())
    ;
}




template <typename MatrixType>
handle<> divideByScalarWithoutCoercion(python::object op1, python::object op2)
{
  python::extract<typename MatrixType::value_type> op2_scalar(op2);
  if (op2_scalar.check())
  {
    const MatrixType &mat = python::extract<MatrixType>(op1);
    return handle_from_new_ptr(new MatrixType(mat / op2_scalar()));
  }

  return handle<>(borrowed(Py_NotImplemented));
}




template <typename MatrixType>
handle<> divideByScalar(python::object op1, python::object op2)
{
  handle<> result = divideByScalarWithoutCoercion<MatrixType>(op1, op2);
  if (result.get() != Py_NotImplemented)
    return result;
  else
    return handle<>(PyObject_CallMethod(op1.ptr(), 
        (char *) "_cast_and_retry", 
        (char *) "sO",
        "div", op2.ptr()));
}




template <typename MatrixType>
handle<> divideByScalarInPlaceWithoutCoercion(python::object op1, python::object op2)
{
  python::extract<typename MatrixType::value_type> op2_scalar(op2);
  if (op2_scalar.check())
  {
    MatrixType &mat = python::extract<MatrixType &>(op1);
    mat /= op2_scalar();
    return handle_from_object(op1);
  }

  return handle<>(borrowed(Py_NotImplemented));
}




template <typename MatrixType>
handle<> divideByScalarInPlace(python::object op1, python::object op2)
{
  handle<> result = divideByScalarInPlaceWithoutCoercion<MatrixType>(op1, op2);
  if (result.get() != Py_NotImplemented)
    return result;
  else
    return handle<>( PyObject_CallMethod(op1.ptr(), 
        (char *) "_cast_and_retry", 
        (char *) "sO",
        "idiv", op2.ptr()));
}




// vector wrapper -------------------------------------------------------------
template <typename VectorType>
handle<> multiplyVectorWithoutCoercion(python::object op1, python::object op2)
{
  typedef typename VectorType::value_type value_t;

  python::extract<VectorType> op2_vec(op2);
  if (op2_vec.check())
  {
    const VectorType &vec = python::extract<VectorType>(op1);
    const VectorType &vec2 = op2_vec();
    return handle_from_rvalue(inner_prod(vec, vec2));
  }

  python::extract<typename VectorType::value_type> op2_scalar(op2);
  if (op2_scalar.check())
  {
    const VectorType &vec = python::extract<VectorType>(op1);
    return handle_from_new_ptr(new VectorType(vec * op2_scalar()));
  }

  return handle<>(borrowed(Py_NotImplemented));
}




template <typename VectorType>
handle<> multiplyVector(python::object op1, python::object op2)
{
  handle<> result = multiplyVectorWithoutCoercion<VectorType>(op1, op2);
  if (result.get() != Py_NotImplemented)
    return result;
  else
    return handle<>(PyObject_CallMethod(op1.ptr(), 
        (char *) "_cast_and_retry", 
        (char *) "sO",
        "mul", op2.ptr()));
}




template <typename VectorType>
handle<> multiplyVectorOuterWithoutCoercion(python::object op1, python::object op2)
{
  typedef typename VectorType::value_type value_type;
  python::extract<VectorType> op2_vec(op2);
  if (op2_vec.check())
  {
    const VectorType &vec = python::extract<VectorType>(op1);
    const VectorType &vec2 = op2_vec();
    return handle_from_new_ptr(new ublas::matrix<value_type>(outer_prod(vec, vec2)));
  }

  return handle<>(borrowed(Py_NotImplemented));
}




template <typename VectorType>
handle<> multiplyVectorOuter(python::object op1, python::object op2)
{
  handle<> result = multiplyVectorOuterWithoutCoercion<VectorType>(op1, op2);
  if (result.get() != Py_NotImplemented)
    return result;
  else
    return handle<>(PyObject_CallMethod(
        op1.ptr(), 
        (char *) "_cast_and_retry", 
        (char *) "sO",
        "_outerproduct", op2.ptr()));
}





template <typename VectorType>
handle<> crossproduct(const VectorType &vec1, const VectorType &vec2)
{
  if (vec1.size() == 3 && vec2.size() == 3)
  {
    std::auto_ptr<VectorType> result(new VectorType(3));

    (*result)[0] = vec1[1]*vec2[2]-vec1[2]*vec2[1];
    (*result)[1] = vec1[2]*vec2[0]-vec1[0]*vec2[2];
    (*result)[2] = vec1[0]*vec2[1]-vec1[1]*vec2[0];

    return handle_from_new_ptr(result.release());
  }
  else if (vec1.size() == 2 && vec2.size() == 2)
  {
    std::auto_ptr<VectorType> result(new VectorType(1));
    (*result)[0] = vec1[1]*vec2[2]-vec1[2]*vec2[1];
    return handle_from_new_ptr(result.release());
  }
  else
    PYTHON_ERROR(ValueError, "cross product requires two vectors of dimensions 2 or 3");
}




template <typename PythonClass, typename WrappedClass>
void exposeVectorConcept(PythonClass &pyclass, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;

  exposeElementWiseBehavior(pyclass, WrappedClass());

  pyclass
    .add_property("H", hermiteVector<WrappedClass>,
        "The complex-conjugate transpose of the Array.")
    .add_property("T", transposeVector<WrappedClass>,
        "The transpose of the Array.")

    // products
    .def("__mul__", multiplyVector<WrappedClass>)
    .def("__rmul__", multiplyVector<WrappedClass>)
    .def("_nocast_mul", multiplyVectorWithoutCoercion<WrappedClass>)

    .def("__div__", divideByScalar<WrappedClass>)
    .def("__truediv__", divideByScalar<WrappedClass>)
    .def("_nocast_div", divideByScalarWithoutCoercion<WrappedClass>)
    .def("__idiv__", divideByScalarInPlace<WrappedClass>)
    .def("_nocast_idiv", divideByScalarInPlaceWithoutCoercion<WrappedClass>)

    .def("_outerproduct", multiplyVectorOuter<WrappedClass>)
    .def("_nocast__outerproduct", multiplyVectorOuterWithoutCoercion<WrappedClass>)
    ;
	 
  exposeUfuncs<WrappedClass>(pyclass);
}




template <typename PythonClass, typename ValueType>
void exposeVectorConvertersForValueType(PythonClass &pyclass, ValueType)
{
  pyclass
    .def(python::init<const ublas::vector<ValueType> &>())
    ;
}




template <typename PythonClass, typename T>
void exposeVectorConverters(PythonClass &pyclass, T)
{
   exposeVectorConvertersForValueType(pyclass, T());
}




template <typename PythonClass, typename T>
void exposeVectorConverters(PythonClass &pyclass, std::complex<T>)
{
  exposeVectorConvertersForValueType(pyclass, T());
  exposeVectorConvertersForValueType(pyclass, std::complex<T>());
}




template <typename WrappedClass>
void exposeVectorType(WrappedClass, const std::string &python_typename, const std::string &python_eltypename)
{
  std::string total_typename = python_typename + python_eltypename;
  class_<WrappedClass, boost::noncopyable> pyclass(total_typename.c_str());

  pyclass
    .def(python::init<typename WrappedClass::size_type>())
    .def("_get_filled_matrix", &getFilledVector<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .staticmethod("_get_filled_matrix")
    ;

  exposeVectorConcept(pyclass, WrappedClass());
  exposeIterator(pyclass, total_typename, WrappedClass());
  exposeVectorConverters(pyclass, typename WrappedClass::value_type());

  def("crossproduct", crossproduct<WrappedClass>);

  exposeUfuncs<WrappedClass>(pyclass);
}




// matrix wrapper -------------------------------------------------------------
template <typename MatrixType>
handle<> multiplyMatrixBase(python::object op1, python::object op2, 
                                    bool reverse)
{
  python::extract<MatrixType> op2_mat(op2);
  if (op2_mat.check())
  {
    const MatrixType &mat = python::extract<MatrixType>(op1);
    const MatrixType &mat2 = op2_mat();
    if (mat.size2() != mat2.size1())
      throw std::runtime_error("matrix sizes don't match");
    if (!reverse)
      return handle_from_new_ptr(new MatrixType(prod(mat, mat2)));
    else
      return handle_from_new_ptr(new MatrixType(prod(mat2, mat)));
  }

  typedef
    typename get_corresponding_vector_type<MatrixType>::type
    vector_t;

  python::extract<vector_t> op2_vec(op2);
  if (op2_vec.check())
  {
    const MatrixType &mat = python::extract<MatrixType>(op1);
    const vector_t &vec = op2_vec();
    if (mat.size2() != vec.size())
      throw std::runtime_error("matrix size doesn't match vector");

    std::auto_ptr<ublas::vector<typename MatrixType::value_type> > result(new
                                                                          ublas::vector<typename MatrixType::value_type>(mat.size1()));
    if (!reverse)
      ublas::axpy_prod(mat, vec, *result);
    else
      ublas::axpy_prod(vec, mat, *result);
    return handle_from_new_ptr(result.release());
  }

  python::extract<typename MatrixType::value_type> op2_scalar(op2);
  if (op2_scalar.check())
  {
    const MatrixType &mat = python::extract<MatrixType>(op1);
    return handle_from_new_ptr(new MatrixType(mat * op2_scalar()));
  }

  return handle<>(borrowed(Py_NotImplemented));
}




template <typename MatrixType>
handle<> multiplyMatrix(python::object op1, python::object op2)
{
  handle<> result = multiplyMatrixBase<MatrixType>(op1, op2, false);
  if (result.get() != Py_NotImplemented)
    return result;
  else
    return handle<>(PyObject_CallMethod(op1.ptr(), 
        (char *) "_cast_and_retry", 
        (char *) "sO",
        "mul", op2.ptr()));
}




template <typename MatrixType>
handle<> multiplyMatrixWithoutCoercion(python::object op1, python::object op2)
{
  return multiplyMatrixBase<MatrixType>(op1, op2, false);
}




template <typename MatrixType>
handle<> rmultiplyMatrix(python::object op1, python::object op2)
{
  handle<> result = multiplyMatrixBase<MatrixType>(op1, op2, true);
  if (result.get() != Py_NotImplemented)
    return result;
  else
    return handle<>(PyObject_CallMethod(op1.ptr(), 
        (char *) "_cast_and_retry", 
        (char *) "sO",
        "rmul", op2.ptr()));
}




template <typename MatrixType>
handle<> rmultiplyMatrixWithoutCoercion(python::object op1, python::object op2)
{
  return multiplyMatrixBase<MatrixType>(op1, op2, true);
}




template <typename MatrixType>
handle<> multiplyMatrixInPlaceWithoutCoercion(python::object op1, python::object op2)
{
  python::extract<MatrixType> op2_mat(op2);
  if (op2_mat.check())
  {
    MatrixType &mat = python::extract<MatrixType &>(op1);
    const MatrixType &mat2 = op2_mat();
    if (mat.size2() != mat2.size1())
      throw std::runtime_error("matrix sizes don't match");

    // FIXME: aliasing!
    mat = prod(mat, mat2);

    return handle_from_object(op1);
  }

  python::extract<typename MatrixType::value_type> op2_scalar(op2);
  if (op2_scalar.check())
  {
    MatrixType &mat = python::extract<MatrixType &>(op1);
    mat *= op2_scalar();
    return handle_from_object(op1);
  }

  return handle<>(borrowed(Py_NotImplemented));
}




template <typename MatrixType>
handle<> multiplyMatrixInPlace(python::object op1, python::object op2)
{
  handle<> result = multiplyMatrixInPlaceWithoutCoercion<MatrixType>(op1, op2);
  if (result.get() != Py_NotImplemented)
    return result;
  else
    return handle<>(PyObject_CallMethod(op1.ptr(), 
        (char *) "_cast_and_retry", 
        (char *) "sO",
        "imul", op2.ptr()));
}




template <typename MatrixType>
void matrixSimplePushBack(MatrixType &m, 
                          typename MatrixType::size_type i,
                          typename MatrixType::size_type j,
                          const typename MatrixType::value_type &el)
{
  m(i, j) = el;
}




template <typename MatrixType>
void matrixSimpleAppendElement(MatrixType &m, 
                               typename MatrixType::size_type i,
                               typename MatrixType::size_type j,
                               const typename MatrixType::value_type &el)
{
  m(i, j) += el;
}




template <typename MatrixType>
void insertElementWrapper(MatrixType &m, 
                   typename MatrixType::size_type i,
                   typename MatrixType::size_type j,
                   const typename MatrixType::value_type &el)
{
  m.insert_element(i, j, el);
}




template <typename WrappedClass, typename SmallMatrix, typename PythonClass>
void exposeAddScattered(PythonClass &pyclass)
{
  using python::arg;

  pyclass
    .def("add_block", addBlock<WrappedClass, SmallMatrix>,
        (arg("self"), arg("start_row"), arg("start_column"), arg("small_mat")),
        "Add C{small_mat} to self, starting at C{start_row,start_column}.")
    .def("add_scattered", addScattered<WrappedClass, SmallMatrix>,
        (arg("self"), arg("row_indices"), arg("column_indices"), arg("small_mat")),
        "Add C{small_mat} at intersections of C{row_indices} and "
        "C{column_indices}.")
    .def("add_scattered_with_skip", addScatteredWithSkip<WrappedClass, SmallMatrix>,
        (arg("self"), arg("row_indices"), arg("column_indices"), arg("small_mat")),
        "Add C{small_mat} at intersections of C{row_indices} and "
        "C{column_indices}. Entries of C{row_indices} or C{column_indices} "
        "may be negative to skip this row or column.")
    ;
}




template <typename PythonClass, typename WrappedClass>
void exposeMatrixConcept(PythonClass &pyclass, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;

  exposeElementWiseBehavior(pyclass, WrappedClass());

  pyclass
    .add_property("H", hermiteMatrix<WrappedClass>,
        "The complex-conjugate transpose of the Array.")
    .add_property("T", transposeMatrix<WrappedClass>,
        "The transpose of the Array.")

    // products
    .def("__mul__", multiplyMatrix<WrappedClass>)
    .def("__rmul__", rmultiplyMatrix<WrappedClass>)
    .def("_nocast_mul", multiplyMatrixWithoutCoercion<WrappedClass>)
    .def("_nocast_rmul", rmultiplyMatrixWithoutCoercion<WrappedClass>)

    .def("__imul__", multiplyMatrixInPlace<WrappedClass>)
    .def("_nocast_imul", multiplyMatrixInPlaceWithoutCoercion<WrappedClass>)
    .def("__div__", divideByScalar<WrappedClass>)
    .def("__truediv__", divideByScalar<WrappedClass>)
    .def("_nocast_div", divideByScalarWithoutCoercion<WrappedClass>)
    .def("__idiv__", divideByScalarInPlace<WrappedClass>)
    .def("_nocast_idiv", divideByScalarInPlaceWithoutCoercion<WrappedClass>)

    .def("solve_lower", solveLower<WrappedClass>,
	 python::return_value_policy<python::manage_new_object>(),
         "Solve A*x=b with this matrix lower-triangular. Return x.")
    .def("solve_upper", solveUpper<WrappedClass>,
	 python::return_value_policy<python::manage_new_object>(),
         "Solve A*x=b with this matrix upper-triangular. Return x.")
    ;
}




template <typename PythonClass>
struct matrix_converter_exposer
{
  PythonClass &m_pyclass;

public:
  matrix_converter_exposer(PythonClass &pyclass)
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




template <typename PYC, typename MT>
void exposeMatrixSpecialties(PYC, MT)
{
}




template <typename PYC, typename VT, typename L, typename A>
void exposeMatrixSpecialties(PYC &pyclass, ublas::matrix<VT, L, A>)
{
  typedef ublas::matrix<VT, L, A> matrix_type;

  pyclass
    .def("set_element", matrixSimplePushBack<matrix_type>,
        "(i,j,x) Set a[i,j] = x.")
    .def("set_element_past_end", matrixSimplePushBack<matrix_type>,
        "(i,j,x) Set a[i,j] = x assuming no element before i,j in lexical ordering.")
    .def("add_element", matrixSimpleAppendElement<matrix_type>,
        "(i,j,x) Set a[i,j] += x.");

  exposeAddScattered<matrix_type, matrix_type>(pyclass);
  exposeUfuncs<matrix_type>(pyclass);
}




template <typename PYC, typename VT, typename L, std::size_t IB, typename IA, typename TA>
void exposeMatrixSpecialties(PYC &pyclass, ublas::compressed_matrix<VT, L, IB, IA, TA>)
{
  typedef ublas::compressed_matrix<VT, L, IB, IA, TA> cl;

  pyclass
    .def("complete_index1_data", &cl::complete_index1_data,
        "Fill up index data of compressed row storage.")
    .def("set_element_past_end", &cl::push_back,
        "(i,j,x) Set a[i,j] = x assuming no element before i,j in lexical ordering.")
    .add_property("nnz", &cl::nnz, 
        "The number of structural nonzeros in the matrix")
    ;

  pyclass
    .def("__add__", wrapBinaryOp<cl, std::plus<cl> >)
    .def("__sub__", wrapBinaryOp<cl, std::minus<cl> >)
    ;
}




template <typename PYC, typename VT, typename L, std::size_t IB, typename IA, typename TA>
void exposeMatrixSpecialties(PYC &pyclass, ublas::coordinate_matrix<VT, L, IB, IA, TA>)
{
  typedef ublas::coordinate_matrix<VT, L, IB, IA, TA> cl;

  pyclass
    .def("sort", &cl::sort,
        "Make sure coordinate representation is sorted.")
    .def("set_element", insertElementWrapper<cl>,
        "(i,j,x) Set a[i,j] = x.")
    .def("set_element_past_end", &cl::push_back,
        "(i,j,x) Set a[i,j] = x assuming no element before i,j in lexical ordering.")
    .def("add_element", &cl::append_element,
        "(i,j,x) Set a[i,j] += x.")
    .add_property("nnz", &cl::nnz, 
        "The number of structural nonzeros in the matrix")
    ;

  exposeAddScattered<cl, ublas::matrix<VT> >(pyclass);
  exposeAddScattered<cl, cl >(pyclass);
  exposeAddScattered<cl, 
    ublas::compressed_matrix<VT, ublas::column_major, 0, 
    ublas::unbounded_array<int> > >(pyclass);

  pyclass
    .def("__add__", wrapBinaryOp<cl, std::plus<cl> >)
    .def("__sub__", wrapBinaryOp<cl, std::minus<cl> >)
    ;
}




template <typename WrappedClass>
void exposeMatrixType(WrappedClass, const std::string &python_typename, const std::string &python_eltypename)
{
  std::string total_typename = python_typename + python_eltypename;
  typedef class_<WrappedClass, boost::noncopyable> wrapper_class;
  wrapper_class pyclass(total_typename.c_str());

  pyclass
    .def(python::init<typename WrappedClass::size_type, 
        typename WrappedClass::size_type>())

    // special constructors
    .def("_get_filled_matrix", &getFilledMatrix<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .staticmethod("_get_filled_matrix")
    ;

  exposeMatrixConcept(pyclass, WrappedClass());
  exposeIterator(pyclass, total_typename, WrappedClass());
  exposeForMatricesConvertibleTo(matrix_converter_exposer<wrapper_class>(pyclass), 
      typename WrappedClass::value_type());
  exposeMatrixSpecialties(pyclass, WrappedClass());
}




#define EXPOSE_ALL_TYPES \
  exposeAll(double(), "Float64"); \
  exposeAll(std::complex<double>(), "Complex64"); \



} // private namespace


// EMACS-FORMAT-TAG
//
// Local Variables:
// mode: C++
// eval: (c-set-style "stroustrup")
// eval: (c-set-offset 'access-label -2)
// eval: (c-set-offset 'inclass '++)
// c-basic-offset: 2
// tab-width: 8
// End:
