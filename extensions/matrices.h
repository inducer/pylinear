#include <complex>
#include <string>
#include <cmath>
#include <functional>

#include <boost/python.hpp>
#include <boost/mpl/bool.hpp>

#include "meta.h"

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <helpers.h>



using boost::python::class_;
using boost::python::enum_;
using boost::python::self;
using boost::python::def;
using helpers::decomplexify;




namespace {
// helpers --------------------------------------------------------------------
template <typename T>
T extractOneTuple(const python::tuple &tup, T)
{
  if (tup.attr("__len__") != 1)
    throw std::runtime_error("expected tuple of length 1");
  return python::extract<T>(tup[0]);
}




template <typename T>
void extractTwoTuple(const python::tuple &tup, T &i, T &j)
{
  if (tup.attr("__len__") != 2)
    throw std::runtime_error("expected tuple of length 2");
  i = python::extract<T>(tup[0]);
  j = python::extract<T>(tup[2]);
}




// shape accessors ------------------------------------------------------------
template <typename MatrixType>
inline unsigned getLength(const MatrixType &m)
{ 
  return m.size1() * m.size2();
}




template <typename ValueType>
inline unsigned getLength(const ublas::vector<ValueType> &m)
{ 
  return m.size();
}




template <typename MatrixType>
inline python::object getShape(const MatrixType &m)
{ 
  return python::make_tuple(m.size1(), m.size2());
}




template <typename ValueType>
inline python::object getShape(const ublas::vector<ValueType> &m)
{ 
  return python::make_tuple(m.size());
}




template <typename MatrixType>
inline void setShape(MatrixType &m, const python::tuple &new_shape)
{ 
  typename MatrixType::size_type h,w;
  extractTwoTuple(new_shape, h, w);
  m.resize(h,w);
}




template <typename ValueType>
inline void setShape(ublas::vector<ValueType> &m, const python::tuple &new_shape)
{ 
  m.resize(extractOneTuple(new_shape, ublas::vector<ValueType>::size_type()));
}




// iterator interface ---------------------------------------------------------
template <typename MatrixType, typename _is_vector = typename is_vector<MatrixType>::type >
struct python_matrix_key_iterator
{
  typename MatrixType::iterator1 m_iterator1;
  typename MatrixType::iterator2 m_iterator2;

  python_matrix_key_iterator *iter()
  {
    return this;
  }

  python::object next()
  {
    if (m_iterator2 == m_iterator1.end())
    {
      if (m_iterator1 == m_iterator1().end1())
      {
        PyErr_SetNone(PyExc_StopIteration);
        throw python::error_already_set();
      }

      m_iterator1++;
      m_iterator2 = m_iterator1.begin();

      if (m_iterator1 == m_iterator1().end1())
      {
        PyErr_SetNone(PyExc_StopIteration);
        throw python::error_already_set();
      }
    }

    python::object result = python::make_tuple(m_iterator2.index1(), m_iterator2.index2());
    m_iterator2++;
    return result;
  }

  static python_matrix_key_iterator *obtain(MatrixType &m)
  {
    std::auto_ptr<python_matrix_key_iterator> it(new python_matrix_key_iterator);
    it->m_iterator1 = m.begin1();
    it->m_iterator2 = it->m_iterator1.begin();
    return it.release();
  }
};




template <typename MatrixType>
struct python_matrix_key_iterator<MatrixType, mpl::true_>
{
  typename MatrixType::iterator m_iterator;

  python_matrix_key_iterator *iter()
  {
    return this;
  }

  unsigned next()
  {
    if (m_iterator == m_iterator().end())
    {
      PyErr_SetNone(PyExc_StopIteration);
      throw python::error_already_set();
    }

    return m_iterator++.index();
  }

  static python_matrix_key_iterator *obtain(MatrixType &m)
  {
    std::auto_ptr<python_matrix_key_iterator> it(new python_matrix_key_iterator);
    it->m_iterator = m.begin();
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

  python::object next()
  {
    if (m_row_index >= m_matrix.size1())
    {
      PyErr_SetNone(PyExc_StopIteration);
      throw python::error_already_set();
    }

    return python::object(
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
  int m_start;
  int m_end;
  int m_step;
  int m_sliceLength;
};




static void translateIndex(PyObject *slice_or_constant, slice_info &si, int my_length)
{
  si.m_was_slice = PySlice_Check(slice_or_constant);
  if (si.m_was_slice)
  {
    PySliceObject *slice = reinterpret_cast<PySliceObject *>(slice_or_constant);
    if (PySlice_GetIndicesEx(slice, my_length, &si.m_start, &si.m_end, 
          &si.m_step, &si.m_sliceLength) != 0)
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
    si.m_step = 1;
    si.m_sliceLength = 1;
  }
  else
    throw std::out_of_range("invalid index object");
}




template <typename MatrixType>
static python::object getElement(const MatrixType &m, PyObject *index)
{ 
  typedef
    typename get_corresponding_vector_type<MatrixType>::type
    vector_t;

  if (PyTuple_Check(index))
  {
    // we have a tuple
    if (PyTuple_GET_SIZE(index) != 2)
      throw std::out_of_range("expected tuple of size 2");

    slice_info si1, si2;
    translateIndex(PyTuple_GET_ITEM(index, 0), si1, m.size1());
    translateIndex(PyTuple_GET_ITEM(index, 1), si2, m.size2());

    if (!si1.m_was_slice && !si2.m_was_slice)
      return python::object(m(si1.m_start, si2.m_start));
    else if (!si1.m_was_slice)
      return python::object(new vector_t(row(m,si1.m_start)));
    else if (!si2.m_was_slice)
      return python::object(new vector_t(column(m,si2.m_start)));
    else
    {
      return python::object(
          new MatrixType(project(m,
              ublas::slice(si1.m_start, si1.m_step, si1.m_sliceLength),
              ublas::slice(si2.m_start, si2.m_step, si2.m_sliceLength))));
    }
  }
  else
  {
    slice_info si;
    translateIndex(index, si, m.size1());

    if (!si.m_was_slice)
      return python::object(
          new vector_t(
            row(m, si.m_start)));
    else
      return python::object(
          new MatrixType(project(m,
              ublas::slice(si.m_start, si.m_step, si.m_sliceLength),
              ublas::slice(0, 1, m.size2())
              )));
  }
}




template <typename ValueType>
static python::object getElement(const ublas::vector<ValueType> &m, PyObject *index)
{ 
  slice_info si;
  translateIndex(index, si, m.size());

  if (!si.m_was_slice)
    return python::object(m(si.m_start));
  else
    return python::object(
        new ublas::vector<ValueType>(project(m, ublas::slice(si.m_start, si.m_step, si.m_sliceLength))));
}




template <typename MatrixType>
static void setElement(MatrixType &m, PyObject *index, python::object &new_value)
{ 
  python::extract<typename MatrixType::value_type> new_scalar(new_value);
  python::extract<const MatrixType &> new_matrix(new_value);

  typedef 
    typename get_corresponding_vector_type<MatrixType>::type
    vector_type;

  python::extract<vector_type> new_vector(new_value);

  if (PyTuple_Check(index))
  {
    // we have a tuple
    if (PyTuple_GET_SIZE(index) != 2)
      throw std::out_of_range("expected tuple of size 2");

    slice_info si1, si2;
    translateIndex(PyTuple_GET_ITEM(index, 0), si1, m.size1());
    translateIndex(PyTuple_GET_ITEM(index, 1), si2, m.size2());

    if (si1.m_sliceLength == 1 && si2.m_sliceLength == 1 && new_scalar.check())
    {
      m(si1.m_start, si2.m_start) = new_scalar();
    }
    else if (si1.m_sliceLength == 1 && new_vector.check())
    {
      vector_type new_vec = new_vector();

      if (new_vec.size() != m.size2())
        throw std::out_of_range("submatrix is wrong size for assignment");

      row(m,si1.m_start) = new_vec;
    }
    else if (si2.m_sliceLength == 1 && new_vector.check())
    {
      vector_type new_vec = new_vector();

      if (new_vec.size() != m.size1())
        throw std::out_of_range("submatrix is wrong size for assignment");

      column(m,si2.m_start) = new_vec;
    }
    else
    {
      const MatrixType &new_mat = new_matrix();
      if (int(new_mat.size1()) != si1.m_sliceLength || int(new_mat.size2()) != si2.m_sliceLength)
        throw std::out_of_range("submatrix is wrong size for assignment");

      project(m,
          ublas::slice(si1.m_start, si1.m_step, si1.m_sliceLength),
          ublas::slice(si2.m_start, si2.m_step, si2.m_sliceLength)) = new_mat;
    }
  }
  else
  {
    slice_info si;
    translateIndex(index, si, m.size1());

    if (si.m_sliceLength == 1 && new_vector.check())
    {
      vector_type new_vec = new_vector();

      if (new_vec.size() != m.size2())
        throw std::out_of_range("submatrix is wrong size for assignment");

      row(m,si.m_start) = new_vec;
    }
    else
    {
      const MatrixType &new_mat = new_matrix();

      if (int(new_mat.size1()) != si.m_sliceLength || new_mat.size2() != m.size2())
        throw std::out_of_range("submatrix is wrong size for assignment");

      project(m,
          ublas::slice(si.m_start, si.m_step, si.m_sliceLength),
          ublas::slice(0, 1, m.size2())) = new_mat();
    }
  }
}




template <typename ValueType>
static void setElement(ublas::vector<ValueType> &m, PyObject *index, python::object &new_value)
{ 
  python::extract<typename ublas::vector<ValueType>::value_type> new_scalar(new_value);
  python::extract<const ublas::vector<ValueType> &> new_matrix(new_value);

  slice_info si;
  translateIndex(index, si, m.size());

  if (si.m_sliceLength == 1 && new_scalar.check())
    m(si.m_start) = new_scalar();
  else
    project(m, ublas::slice(si.m_start, si.m_step, si.m_sliceLength)) = 
      new_matrix();
}




// specialty constructors -----------------------------------------------------
template <typename MatrixType>
static MatrixType *getIdentityMatrix(unsigned n)
{
  return new MatrixType(
      ublas::identity_matrix<typename MatrixType::value_type>(n));
}




template <typename MatrixType>
static MatrixType *getFilledMatrix(
    typename MatrixType::size_type size1, 
    typename MatrixType::size_type size2, 
    const typename MatrixType::value_type &value)
{
  std::auto_ptr<MatrixType> mat(new MatrixType(size1, size2));
  
  // cannot use iterators here since sparse types are initially "empty"
  for (unsigned i = 0; i < mat->size1(); i++)
    for (unsigned j = 0; j < mat->size1(); j++)
      (*mat)(i,j) = value;

  return mat.release();
}




template <typename MatrixType>
static MatrixType *getFilledVector(
    typename MatrixType::size_type size1, 
    const typename MatrixType::value_type &value)
{
  std::auto_ptr<MatrixType> mat(new MatrixType(size1));

  // cannot use iterators here since sparse types are initially "empty"
  for (unsigned i = 0; i < mat->size(); i++)
    (*mat)(i) = value;

  return mat.release();
}




// operators ------------------------------------------------------------------
template <typename MatrixType>
MatrixType negateOp(const MatrixType &m) { return -m; }

template <typename MatrixType>
MatrixType plusOp(const MatrixType &m1, const MatrixType &m2) { return m1+m2; }

template <typename MatrixType>
MatrixType minusOp(const MatrixType &m1, const MatrixType &m2) { return m1-m2; }

template <typename MatrixType>
MatrixType plusAssignOp(MatrixType &m1, const MatrixType &m2) { return m1 += m2; }

template <typename MatrixType>
MatrixType minusAssignOp(MatrixType &m1, const MatrixType &m2) { return m1 -= m2; }

template <typename MatrixType, typename Scalar>
MatrixType scalarTimesOp(const MatrixType &m1, const Scalar &s) { return m1 * s; }

template <typename MatrixType, typename Scalar>
MatrixType scalarDivideOp(const MatrixType &m1, const Scalar &s) { return m1 / s; }

template <typename MatrixType, typename Scalar>
MatrixType scalarTimesAssignOp(MatrixType &m1, const Scalar &s) { return m1 *= s; }

template <typename MatrixType, typename Scalar>
MatrixType scalarDivideAssignOp(MatrixType &m1, const Scalar &s) { return m1 /= s; }




// universal functions --------------------------------------------------------
template <typename MatrixType>
inline MatrixType *copyNew(const MatrixType &m)
{
  return new MatrixType(m);
}




template <typename Op1, typename Op2>
struct prodMatMatWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_value_type;
  typedef 
    typename ublas::matrix<result_value_type> 
    result_type;

  inline static result_type *apply(const Op1 &op1, const Op2 &op2)
  {
    return new result_type(prod(op1, op2));
  }
};




template <typename Op1, typename Op2>
struct prodMatVecWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_value_type;
  typedef 
    typename ublas::vector<result_value_type> 
    result_type;

  inline static result_type *apply(const Op1 &op1, const Op2 &op2)
  {
    return new result_type(prod(op1, op2));
  }
};




template <typename Op1, typename Op2>
struct inner_prodWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_type;

  inline static result_type apply(const Op1 &op1, const Op2 &op2)
  {
    return inner_prod(op1, op2);
  }
};




template <typename Op1, typename Op2>
struct outer_prodWrapper
{
  typedef 
    typename value_type_promotion::bigger_type<typename Op1::value_type, typename Op2::value_type>::type
    result_value_type;
  typedef 
    typename ublas::matrix<result_value_type> 
    result_type;

  inline static result_type *apply(const Op1 &op1, const Op2 &op2)
  {
    return new result_type(outer_prod(op1, op2));
  }
};




template <typename MatrixType>
static MatrixType *transposeMatrix(const MatrixType &m)
{
  return new MatrixType(trans(m));
}




template <typename MatrixType>
static MatrixType *hermiteMatrix(const MatrixType &m)
{
  return new MatrixType(herm(m));
}




template <typename MatrixType>
struct realWrapper
{
  typedef 
    typename change_value_type<MatrixType, 
      typename decomplexify<typename MatrixType::value_type>::type>::type
    result_type;

  inline static result_type *apply(const MatrixType &m)
  {
    return new result_type(real(m));
  }
};




template <typename MatrixType>
struct imagWrapper
{
  typedef 
    typename change_value_type<MatrixType, 
      typename decomplexify<typename MatrixType::value_type>::type>::type
    result_type;

  inline static result_type *apply(const MatrixType &m)
  {
    return new result_type(imag(m));
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




template <typename T>
inline std::string stringify(const T &obj)
{
  std::stringstream stream;
  stream << obj;
  return stream.rdbuf()->str();
}




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



  template <typename MatrixType, typename Function, typename _is_vector = typename is_vector<MatrixType>::type>
  struct unary_ufunc_applicator
  {
    typedef 
      typename change_value_type<MatrixType, typename Function::result_type>::type
      result_type;

    static result_type *apply(const MatrixType &m)
    {
      Function f;

      std::auto_ptr<result_type> new_mat(new result_type(m.size1(), m.size2()));

      typedef typename MatrixType::const_iterator1 it1_t;
      typedef typename MatrixType::const_iterator2 it2_t;
      for (it1_t it1 = m.begin1(); it1 != m.end1(); ++it1) 
        for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
          new_mat->insert(it2.index1(), it2.index2(), f(*it2));

      return new_mat.release();
    }
  };




  template <typename MatrixType, typename Function>
  struct unary_ufunc_applicator<MatrixType, Function, mpl::true_>
  {
    typedef 
      typename change_value_type<MatrixType, typename Function::result_type>::type
      result_type;

    static result_type *apply(const MatrixType &m)
    {
      Function f;

      std::auto_ptr<result_type> new_mat(new result_type(m. size()));

      typedef typename MatrixType::const_iterator it_t;
      for (it_t it = m.begin(); it != m.end(); ++it) 
        new_mat->insert(it.index(), f(*it));

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
      Function f;
      return f(a1, a2);
    }
  };




  template <typename MatrixType, typename Function, typename _is_vector = typename is_vector<MatrixType>::type>
  struct binary_ufunc_applicator
  {
    template <typename RealFunction>
    static MatrixType *applyBackend(RealFunction f, const MatrixType &m1, python::object obj)
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
            new_mat->insert(it2.index1(), it2.index2(), f(*it2, m2(it2.index1(), it2.index2())));
      }
      else if (v2_extractor.check())
      {
        const vector_type &v2 = v2_extractor();

        if (m1.size1() != v2.size())
          throw std::runtime_error("cannot apply binary ufunc to arrays of different sizes");

        for (it1_t it1 = m1.begin1(); it1 != m1.end1(); ++it1) 
          for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
            new_mat->insert(it2.index1(), it2.index2(), f(*it2, v2(it2.index1())));
      }
      else if (s2_extractor.check())
      {
        value_type s2 = s2_extractor();

        for (it1_t it1 = m1.begin1(); it1 != m1.end1(); ++it1) 
          for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
            new_mat->insert(it2.index1(), it2.index2(), f(*it2, s2));
      }
      else if (n2_extractor.check())
      {
        value_type n2 = n2_extractor();

        for (it1_t it1 = m1.begin1(); it1 != m1.end1(); ++it1) 
          for (it2_t it2 = it1.begin(); it2 != it1.end(); ++it2) 
            new_mat->insert(it2.index1(), it2.index2(), f(*it2, n2));
      }

      return new_mat.release();
    }

    static MatrixType *apply(const MatrixType &m1, python::object obj)
    {
      return applyBackend(Function(), m1, obj);
    }

    static MatrixType *applyReversed(python::object obj, const MatrixType &m2)
    {
      return applyBackend(reverse_binary_function<Function>(), m2, obj);
    }
  };




  template <typename MatrixType, typename Function>
  struct binary_ufunc_applicator<MatrixType, Function, mpl::true_>
  {
    template <typename RealFunction>
    static MatrixType *applyBackend(RealFunction f, const MatrixType &m1, python::object obj)
    {
      typedef 
        typename MatrixType::value_type
        value_type;
      typedef 
        typename decomplexify<value_type>::type
        nc_value_type;
      typedef 
        typename MatrixType::const_iterator
        it_t;

      python::extract<const MatrixType &> m2_extractor(obj);
      python::extract<value_type> s2_extractor(obj);
      python::extract<nc_value_type> n2_extractor(obj);

      std::auto_ptr<MatrixType> new_mat(new MatrixType(m1.size()));

      if (m2_extractor.check())
      {
        const MatrixType &m2 = m2_extractor();

        if (m1.size() != m2.size())
          throw std::runtime_error("cannot apply binary ufunc to vectors of different sizes");

        for (it_t it = m1.begin(); it != m1.end(); ++it) 
          new_mat->insert(it.index(), f(*it, m2(it.index())));
      }
      else if (s2_extractor.check())
      {
        value_type s2 = s2_extractor();

        for (it_t it = m1.begin(); it != m1.end(); ++it) 
          new_mat->insert(it.index(), f(*it, s2));
      }
      else if (n2_extractor.check())
      {
        value_type n2 = n2_extractor();

        for (it_t it = m1.begin(); it != m1.end(); ++it) 
          new_mat->insert(it.index(), f(*it, n2));
      }

      return new_mat.release();
    }

    static MatrixType *apply(const MatrixType &m1, python::object obj)
    {
      return applyBackend(Function(), m1, obj);
    }

    static MatrixType *applyReversed(python::object obj, const MatrixType &m2)
    {
      return applyBackend(reverse_binary_function<Function>(), m2, obj);
    }
  };
}




// wrapper for stuff that is common to vectors and matrices -------------------
template <typename PythonClass, typename WrappedClass>
static void exposeUfuncs(PythonClass &pyc, WrappedClass)
{
  typedef
    typename WrappedClass::value_type
    value_type;

  def("conjugate", conjugateWrapper<WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
  def("real", realWrapper<WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
  def("imaginary", imagWrapper<WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());

#define MAKE_UNARY_UFUNC(f) \
  def(#f, ufuncs::unary_ufunc_applicator<WrappedClass, \
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
  def(NAME, ufuncs::binary_ufunc_applicator<WrappedClass, \
      f<value_type> >::apply, \
      python::return_value_policy<python::manage_new_object>()); \
  def(NAME, ufuncs::binary_ufunc_applicator<WrappedClass, \
      f<value_type> >::applyReversed, \
      python::return_value_policy<python::manage_new_object>());
  MAKE_BINARY_UFUNC("add", std::plus);
  MAKE_BINARY_UFUNC("subtract", std::minus);
  MAKE_BINARY_UFUNC("multiply", std::multiplies);
  MAKE_BINARY_UFUNC("divide", std::divides);
  MAKE_BINARY_UFUNC("divide_safe", std::divides); // FIXME: bogus
  MAKE_BINARY_UFUNC("power", ufuncs::power);
  MAKE_BINARY_UFUNC("maximum", ufuncs::maximum);
  MAKE_BINARY_UFUNC("minimum", ufuncs::minimum);
#undef MAKE_BINARY_UFUNC
}




template <typename PythonClass, typename WrappedClass>
static void exposeElementWiseBehavior(PythonClass &pyc, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;
  pyc
    .def("typecode", &typecode<WrappedClass>)
    .def("copy", copyNew<WrappedClass>, 
        python::return_value_policy<python::manage_new_object>())

    .add_property("shape", 
        (python::object (*)(const WrappedClass &)) getShape, 
        (void (*)(WrappedClass &, const python::tuple &)) setShape)
    .def("__len__", (unsigned (*)(const WrappedClass &)) getLength)
    .def("swap", &WrappedClass::swap)

    .def("__getitem__", (python::object (*)(const WrappedClass &, PyObject *)) getElement)
    .def("__setitem__", (void (*)(WrappedClass &, PyObject *, python::object &)) setElement)

    // stringification
    .def("__str__", &stringify<WrappedClass>)

    // unary negation
    .def("__neg__", negateOp<WrappedClass>)

    // matrix - matrix
    .def("__add__", plusOp<WrappedClass>)
    .def("__sub__", minusOp<WrappedClass>)
    .def("__iadd__", plusAssignOp<WrappedClass>)
    .def("__isub__", minusAssignOp<WrappedClass>)

    // scalar - matrix

    // this is redundant for doubles, but that's faster to compile than yet
    // another partial specialization. Boost.Python doesn't mind
    // double definitions. FIXME: Performance hit?
    .def("__mul__", scalarTimesOp<WrappedClass, double>)
    .def("__mul__", scalarTimesOp<WrappedClass, value_type>)

    .def("__rmul__", scalarTimesOp<WrappedClass, double>)
    .def("__rmul__", scalarTimesOp<WrappedClass, value_type>)

    .def("__div__", scalarDivideOp<WrappedClass, double>)
    .def("__div__", scalarDivideOp<WrappedClass, value_type>)

    .def("__imul__", scalarTimesAssignOp<WrappedClass, double>)
    .def("__imul__", scalarTimesAssignOp<WrappedClass, value_type>)

    .def("__idiv__", scalarDivideAssignOp<WrappedClass, double>)
    .def("__idiv__", scalarDivideAssignOp<WrappedClass, value_type>)
    ;

  exposeUfuncs(pyc, WrappedClass());
}




template <typename PythonClass, typename WrappedClass>
static void exposeIterator(PythonClass &pyc, const std::string &python_typename, WrappedClass)
{
  typedef 
    python_matrix_value_iterator<WrappedClass>
    value_iterator;

  typedef 
    python_matrix_key_iterator<WrappedClass>
    key_iterator;

  pyc
    .def("__iter__", &value_iterator::obtain,
        python::return_value_policy<python::manage_new_object,
        python::return_internal_reference<> >())
    .def("indices", &key_iterator::obtain,
        python::return_value_policy<python::manage_new_object,
        python::return_internal_reference<> >())
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




// vector wrapper -------------------------------------------------------------
template <typename PythonClass, typename WrappedClass>
static void exposeVectorConcept(PythonClass &pyclass, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;

  exposeElementWiseBehavior(pyclass, WrappedClass());

  // inner and outer products
  def("innerproduct", inner_prodWrapper<WrappedClass, WrappedClass>::apply);
  def("outerproduct", outer_prodWrapper<WrappedClass, WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
}




template <typename PythonClass, typename ValueType>
static void exposeVectorConvertersForValueType(PythonClass &pyclass, ValueType)
{
  pyclass
    .def(python::init<const ublas::vector<ValueType> &>())
    ;
}




template <typename PythonClass, typename T>
static void exposeVectorConverters(PythonClass &pyclass, T)
{
  exposeVectorConvertersForValueType(pyclass, T());
}




template <typename PythonClass, typename T>
static void exposeVectorConverters(PythonClass &pyclass, std::complex<T>)
{
  exposeVectorConvertersForValueType(pyclass, T());
  exposeVectorConvertersForValueType(pyclass, std::complex<T>());
}




template <typename WrappedClass>
static void exposeVectorType(WrappedClass, const std::string &python_typename, const std::string &python_eltypename)
{
  std::string total_typename = python_typename + python_eltypename;
  class_<WrappedClass> pyclass(total_typename.c_str());

  pyclass
    .def(python::init<typename WrappedClass::size_type>())
    .def("getFilledMatrix", &getFilledVector<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .staticmethod("getFilledMatrix")
    ;

  exposeVectorConcept(pyclass, WrappedClass());
  exposeIterator(pyclass, total_typename, WrappedClass());
  exposeVectorConverters(pyclass, typename WrappedClass::value_type());
}




// matrix wrapper -------------------------------------------------------------
template <typename PythonClass, typename WrappedClass>
static void exposeMatrixConcept(PythonClass &pyclass, WrappedClass)
{
  typedef typename WrappedClass::value_type value_type;

  exposeElementWiseBehavior(pyclass, WrappedClass());

  // products
  def("matrixmultiply", prodMatMatWrapper<WrappedClass, WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());
  def("matrixmultiply", prodMatVecWrapper<WrappedClass, ublas::vector<value_type> >::apply,
      python::return_value_policy<python::manage_new_object>());
  def("matrixmultiply", prodMatVecWrapper<ublas::vector<value_type>, WrappedClass>::apply,
      python::return_value_policy<python::manage_new_object>());

  def("transpose", transposeMatrix<WrappedClass>,
      python::return_value_policy<python::manage_new_object>());
  def("hermite", hermiteMatrix<WrappedClass>,
      python::return_value_policy<python::manage_new_object>());
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




template <typename WrappedClass>
static void exposeMatrixType(WrappedClass, const std::string &python_typename, const std::string &python_eltypename)
{
  std::string total_typename = python_typename + python_eltypename;
  class_<WrappedClass> pyclass(total_typename.c_str());

  pyclass
    .def(python::init<typename WrappedClass::size_type, 
        typename WrappedClass::size_type>())

    // special constructors
    .def("getIdentityMatrix", &getIdentityMatrix<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .def("getFilledMatrix", &getFilledMatrix<WrappedClass>,
        python::return_value_policy<python::manage_new_object>())
    .staticmethod("getIdentityMatrix")
    .staticmethod("getFilledMatrix")
    ;

  exposeMatrixConcept(pyclass, WrappedClass());
  exposeIterator(pyclass, total_typename, WrappedClass());
  exposeForMatricesConvertibleTo(matrix_converter_exposer<class_<WrappedClass> >(pyclass), 
      typename WrappedClass::value_type());
}




#define EXPOSE_ALL_TYPES \
  exposeAll(double(), "Float64"); \
  exposeAll(std::complex<double>(), "Complex64"); \



} // private namespace
