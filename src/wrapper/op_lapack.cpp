//
// Copyright (c) 2004-2007
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


#include <algorithm>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/type.hpp>

// tools ----------------------------------------------------------------------
#include "meta.hpp"
#include "python_helpers.hpp"

// lapack ---------------------------------------------------------------------
#ifdef USE_LAPACK
#include <boost/numeric/bindings/lapack/gesdd.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/lapack/heev.hpp>
#include <boost/numeric/bindings/lapack/geev.hpp>
#include <boost/numeric/bindings/lapack/gesv.hpp>
#endif // USE_LAPACK




namespace lapack = boost::numeric::bindings::lapack;
namespace 
{
// permutations ---------------------------------------------------------------
/*
Compute sign of permutation `p` by counting the number of
interchanges required to change the given permutation into the
identity one.

Algorithm from http://people.scs.fsu.edu/~burkardt/math2071/perm_sign.m
*/
template <class Container>
int permutation_sign(Container &p)
{
  int n = p.size();
  int s = +1;

  for (int i = 0; i < n; i++)
  {
    // J is the current position of item I.
    int j = i;

    while (p[j] != i)
      j++;

    // Unless the item is already in the correct place, restore it.
    if (j != i)
    {
      std::swap(p[i], p[j]);
      s = -s;
    }
  }
  return s;
}




int permutation_sign_wrapper(boost::python::list py_permut)
{
  std::vector<int> permut;
  copy(
      boost::python::stl_input_iterator<int>(py_permut),
      boost::python::stl_input_iterator<int>(),
      back_inserter(permut));
  return permutation_sign(permut);
}




// svd ------------------------------------------------------------------------
#ifdef USE_LAPACK
template <typename ValueType>
static PyObject *svd_wrapper(const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> mat;
  typedef ublas::matrix<ValueType, ublas::column_major> fortran_mat;
  
  fortran_mat a_copy(a), u(a.size1(), a.size1()), vt(a.size2(), a.size2());
  ublas::vector<double> s(std::min(a.size1(), a.size2()));
  
  int ierr = lapack::gesdd(a_copy, s, u, vt);
  if (ierr < 0)
    throw std::runtime_error("invalid argument to gesdd");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");

  typedef ublas::matrix<ValueType> mat;
  typedef ublas::vector<ValueType> vec;
  return Py_BuildValue("(NNN)", 
                       pyobject_from_new_ptr(new mat(u)),
                       pyobject_from_new_ptr(new vec(s)), 
                       pyobject_from_new_ptr(new mat(vt)));
}
#endif // USE_LAPACK




// eigenvectors ---------------------------------------------------------------
#ifdef USE_LAPACK
void _Heigenvectors_backend(char jobz, char uplo, 
			   ublas::matrix<double, ublas::column_major> &a, 
			   ublas::vector<double> &w) 
{
  int ierr = lapack::syev(jobz, uplo, a, w, lapack::optimal_workspace());
  if (ierr < 0)
    throw std::runtime_error("invalid argument to syev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");
}

void _Heigenvectors_backend(char jobz, char uplo, 
			    ublas::matrix<std::complex<double>, ublas::column_major> &a, 
			    ublas::vector<double> &w) 
{
  int ierr = lapack::heev(jobz, uplo, a, w, lapack::optimal_workspace());
  if (ierr < 0)
    throw std::runtime_error("invalid argument to heev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");
}

template <typename ValueType>
static PyObject *Heigenvectors_wrapper(bool get_vectors, bool upper, 
                                      const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> mat;
  typedef ublas::matrix<ValueType, ublas::column_major> fortran_mat;
  typedef ublas::vector<double> vec;
  
  fortran_mat a_copy(a);
  std::auto_ptr<vec> w(new vec(a.size1()));
  
  _Heigenvectors_backend(get_vectors ? 'V' : 'N',
			 upper ? 'U' : 'L',
			 a_copy, *w);

  typedef ublas::matrix<ValueType> mat;
  return Py_BuildValue("(NN)",
                       pyobject_from_new_ptr(new mat(a_copy)), 
                       pyobject_from_new_ptr(w.release()));
}




template <typename ValueType>
static PyObject *eigenvectors_wrapper(unsigned get_left_vectors, 
                                     unsigned get_right_vectors,
                                     const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> mat;
  typedef ublas::matrix<ValueType, ublas::column_major> fortran_mat;
  typedef ublas::vector<typename helpers::complexify<ValueType>::type > eval_vec;
  typedef ublas::matrix<typename helpers::complexify<ValueType>::type, ublas::column_major> evec_mat;
  
  int const n = a.size1();
  fortran_mat a_copy(a);
  std::auto_ptr<eval_vec> w(new eval_vec(a.size1()));

  std::auto_ptr<evec_mat> vl, vr;
  if (get_left_vectors)
    vl = std::auto_ptr<evec_mat>(new evec_mat(n, n));
  if (get_right_vectors)
    vr = std::auto_ptr<evec_mat>(new evec_mat(n, n));

  int ierr = lapack::geev(a_copy, *w, vl.get(), vr.get(),
      lapack::optimal_workspace());

  if (ierr < 0)
    throw std::runtime_error("invalid argument to geev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");

  typedef ublas::matrix<typename helpers::complexify<ValueType>::type> cmat;
  typedef ublas::vector<ValueType> vec;

  if (get_left_vectors)
  {
    if (get_right_vectors)
      return Py_BuildValue("(NNN)",
                           pyobject_from_new_ptr(w.release()),
                           pyobject_from_new_ptr(new cmat(*vl)),
                           pyobject_from_new_ptr(new cmat(*vr)));
    else
      return Py_BuildValue("(NNs)",
                           pyobject_from_new_ptr(w.release()),
                           pyobject_from_new_ptr(new cmat(*vl)),
                           NULL);
  }
  else
  {
    if (get_right_vectors)
      return Py_BuildValue("(NsN)",
                           pyobject_from_new_ptr(w.release()),
                           NULL,
                           pyobject_from_new_ptr(new cmat(*vr)));
    else
      return Py_BuildValue("(Nss)",
                           pyobject_from_new_ptr(w.release()),
                           NULL,
                           NULL);
  }
}




// lu -------------------------------------------------------------------------
template <typename ValueType>
PyObject *lu_wrapper(const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> matrix_t;
  typedef ublas::matrix<ValueType, ublas::column_major> col_matrix_t;

  const unsigned piv_len = std::min(a.size1(), a.size2());

  col_matrix_t temp(a);
  ublas::vector<int> piv(piv_len);

  int info = lapack::getrf(temp, piv);
  if (info < 0)
    throw std::runtime_error("invalid argument to Xgetrf");
  
  std::auto_ptr<matrix_t> l(new matrix_t(a.size1(), a.size2()));
  l->clear();
  std::auto_ptr<matrix_t> u(new matrix_t(a.size1(), a.size2()));
  u->clear();

  for (unsigned i = 0; i < a.size1(); i++)
  {
    unsigned j = 0;
    for (; j < std::min(i, a.size2()); j++) (*l)(i,j) = temp(i,j);
    (*l)(i,i) = 1;
    for (; j < a.size2(); j++) (*u)(i,j) = temp(i,j);
  }

  ublas::vector<int> permut(piv_len);
  for (unsigned i = 0; i < piv_len; i++) 
    permut[i] = i;
  for (unsigned i = 0; i < piv_len; i++) 
    std::swap(permut[i], permut[piv[i]-1]);

  python::list py_permut;
  for (unsigned i = 0; i < piv_len; i++)
    py_permut.append(permut[i]);
  
  return Py_BuildValue("(NNOi)",
                       pyobject_from_new_ptr(l.release()), 
                       pyobject_from_new_ptr(u.release()), 
                       py_permut.ptr(),
                       permutation_sign(permut));
}




template <typename ValueType>
ValueType determinant(const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> matrix_t;
  typedef ublas::matrix<ValueType, ublas::column_major> col_matrix_t;

  if (a.size1() != a.size2())
    throw std::runtime_error("determinant of non-square matrices is not defined");

  const unsigned n = a.size1();

  col_matrix_t temp(a);
  ublas::vector<int> piv(n);

  int info = lapack::getrf(temp, piv);
  if (info < 0)
    throw std::runtime_error("invalid argument to Xgetrf");
  if (info > 0)
    return 0;
  
  ublas::vector<int> permut(n);
  for (unsigned i = 0; i < n; i++) 
    permut[i] = i;
  for (unsigned i = 0; i < n; i++) 
    std::swap(permut[i], permut[piv[i]-1]);

  ValueType result = permutation_sign(permut);

  for (unsigned i = 0; i < n; i++)
    result *= temp(i,i);

  return result;
}
#endif // USE_LAPACK




// solve linear system --------------------------------------------------------
template <typename ValueType>
PyObject *solve_linear_system_with_matrix_wrapper(
    const ublas::matrix<ValueType> &a,
    const ublas::matrix<ValueType> &b)
{
  typedef ublas::matrix<ValueType> matrix_t;
  typedef ublas::matrix<ValueType, ublas::column_major> col_matrix_t;

  if (a.size1() != a.size2())
    throw std::runtime_error("linear solve requires square matrix");

  col_matrix_t a_temp(a);
  col_matrix_t x_temp(b);

  int info = lapack::gesv(a_temp, x_temp);
  if (info < 0)
    throw std::runtime_error("invalid argument to Xgesv");
  if (info > 0)
    throw std::runtime_error("matrix singular in linear solve");
  
  std::auto_ptr<matrix_t> x(new matrix_t(x_temp));
  return pyobject_from_new_ptr(x.release());
}




template <typename ValueType>
PyObject *solve_linear_system_with_vector_wrapper(
    const ublas::matrix<ValueType> &a,
    const ublas::vector<ValueType> &b)
{
  typedef ublas::matrix<ValueType> matrix_t;
  typedef ublas::matrix<ValueType, ublas::column_major> col_matrix_t;
  typedef ublas::vector<ValueType> vector_t;

  if (a.size1() != a.size2())
    throw std::runtime_error("linear solve requires square matrix");
  if (a.size1() != b.size())
    throw std::runtime_error("linear solve requires that the vector matches the matrix");
  const unsigned n = a.size1();

  col_matrix_t a_temp(a);
  col_matrix_t x_temp(n, 1);
  column(x_temp, 0) = b;

  int info = lapack::gesv(a_temp, x_temp);

  if (info < 0)
    throw std::runtime_error("invalid argument to Xgesv");
  if (info > 0)
    throw std::runtime_error("matrix singular in linear solve");
  
  std::auto_ptr<vector_t> x(new vector_t(column(x_temp, 0)));
  return pyobject_from_new_ptr(x.release());
}




// ----------------------------------------------------------------------------




template <typename ValueType>
void exposeLapackAlgorithms(ValueType)
{
#ifdef USE_LAPACK
  python::def("singular_value_decomposition", svd_wrapper<ValueType>);
  python::def("Heigenvectors", Heigenvectors_wrapper<ValueType>);
  python::def("eigenvectors", eigenvectors_wrapper<ValueType>);
  python::def("lu_lapack", lu_wrapper<ValueType>);
  python::def("det_lapack", determinant<ValueType>);
  python::def("solve_linear_system", 
      solve_linear_system_with_matrix_wrapper<ValueType>);
  python::def("solve_linear_system", 
      solve_linear_system_with_vector_wrapper<ValueType>);
#endif // USE_LAPACK
  python::def("permutation_sign", permutation_sign_wrapper);
}

} // anonymous namespace






void expose_lapack()
{
  exposeLapackAlgorithms(double());
  exposeLapackAlgorithms(std::complex<double>());
}
