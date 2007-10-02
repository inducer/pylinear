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


#include <boost/python.hpp>
#include <cg.hpp>
#include <bicgstab.hpp>
#include <lu.hpp>
#include <cholesky.hpp>

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
#endif // USE_LAPACK




namespace 
{
// svd ------------------------------------------------------------------------
#ifdef USE_LAPACK
template <typename ValueType>
static PyObject *svdWrapper(const ublas::matrix<ValueType> &a)
{
  typedef ublas::matrix<ValueType> mat;
  typedef ublas::matrix<ValueType, ublas::column_major> fortran_mat;
  
  fortran_mat a_copy(a), u(a.size1(), a.size1()), vt(a.size2(), a.size2());
  ublas::vector<double> s(std::min(a.size1(), a.size2()));
  
  int ierr = boost::numeric::bindings::lapack::gesdd(a_copy, s, u, vt);
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
  int ierr = boost::numeric::bindings::lapack::syev(jobz, uplo, a, w, 
						boost::numeric::bindings::lapack::optimal_workspace());
  if (ierr < 0)
    throw std::runtime_error("invalid argument to syev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");
}

void _Heigenvectors_backend(char jobz, char uplo, 
			    ublas::matrix<std::complex<double>, ublas::column_major> &a, 
			    ublas::vector<double> &w) 
{
  int ierr = boost::numeric::bindings::lapack::heev(jobz, uplo, a, w, 
						    boost::numeric::bindings::lapack::optimal_workspace());
  if (ierr < 0)
    throw std::runtime_error("invalid argument to heev");
  else if (ierr > 0)
    throw std::runtime_error("no convergence for given matrix");
}

template <typename ValueType>
static PyObject *HeigenvectorsWrapper(bool get_vectors, bool upper, 
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
static PyObject *eigenvectorsWrapper(unsigned get_left_vectors, 
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

  int ierr = boost::numeric::bindings::lapack::geev(a_copy, *w, vl.get(), vr.get(),
						    boost::numeric::bindings::lapack::optimal_workspace());

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
#endif // USE_LAPACK




// ----------------------------------------------------------------------------
template <typename ValueType>
void exposeLapackAlgorithms(ValueType)
{
#ifdef USE_LAPACK
  python::def("singular_value_decomposition", svdWrapper<ValueType>);
  python::def("Heigenvectors", HeigenvectorsWrapper<ValueType>);
  python::def("eigenvectors", eigenvectorsWrapper<ValueType>);
#endif // USE_LAPACK
}

} // anonymous namespace






void expose_lapack()
{
  exposeLapackAlgorithms(double());
  exposeLapackAlgorithms(std::complex<double>());
}
