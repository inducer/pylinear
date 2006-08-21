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




#ifndef HEADER_SEEN_MANAGED_ADAPTORS_HPP
#define HEADER_SEEN_MANAGED_ADAPTORS_HPP




#include <memory>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/utility/base_from_member.hpp>




template <typename AdaptedMatrix>
class managed_symmetric_adaptor : 
private boost::base_from_member<std::auto_ptr<AdaptedMatrix> >,
  public boost::numeric::ublas::symmetric_adaptor<AdaptedMatrix>
{
  typedef boost::numeric::ublas::symmetric_adaptor<AdaptedMatrix> super;
  typedef boost::base_from_member<std::auto_ptr<AdaptedMatrix> > m_data;
  public:
  typedef AdaptedMatrix adapted_matrix;
  typedef typename super::size_type size_type;
  typedef typename super::value_type value_type;

  managed_symmetric_adaptor()
    : m_data(new AdaptedMatrix()), super(*m_data::member)
    {
    }

  managed_symmetric_adaptor(size_type s1, size_type s2)
    : m_data(new AdaptedMatrix(s1, s1)), super(*m_data::member)
    {
      if (s1 != s2)
        throw std::runtime_error( "symmetric matrices are quadratic" );
    }

  managed_symmetric_adaptor(const managed_symmetric_adaptor &m)
    : m_data(new AdaptedMatrix(*m.member)), super(*m_data::member)
    {
    }

  template <class AE>
    managed_symmetric_adaptor(const boost::numeric::ublas::matrix_expression<AE> &ae)
    : m_data(new AdaptedMatrix(ae().size1(), ae().size1())), super(*m_data::member)
    {
      if (ae().size1() != ae().size2())
        throw std::runtime_error( "symmetric matrices are quadratic" );
      super::operator=(ae);
    }

  void resize(size_type s1, size_type s2)
  {
    if (s1 != s2)
      throw std::runtime_error( "symmetric matrices are quadratic" );
    m_data::member->resize(s1, s2);
  }

  void insert(size_type i1, size_type i2, const value_type &val)
  {
    super::operator()(i1, i2) = val;
  }

  void push_back(size_type i1, size_type i2, const value_type &val)
  {
    super::operator()(i1, i2) = val;
  }

  managed_symmetric_adaptor &operator=(const managed_symmetric_adaptor &m)
  {
    *m_data::member.operator=(*m.member);
  }

  template <class AE>
    managed_symmetric_adaptor &operator=(const boost::numeric::ublas::matrix_expression<AE> &ae)
    {
      if (ae().size1() != ae().size2())
        throw std::runtime_error( "symmetric matrices are quadratic" );
      m_data::member->resize(ae.size1(), ae.size2());
      super::operator=(ae);
    }
};




template <typename AdaptedMatrix>
class managed_hermitian_adaptor : 
private boost::base_from_member<std::auto_ptr<AdaptedMatrix> >,
  public boost::numeric::ublas::hermitian_adaptor<AdaptedMatrix>
{
  typedef boost::numeric::ublas::hermitian_adaptor<AdaptedMatrix> super;
  typedef boost::base_from_member<std::auto_ptr<AdaptedMatrix> > m_data;
  public:
  typedef AdaptedMatrix adapted_matrix;
  typedef typename super::size_type size_type;
  typedef typename super::value_type value_type;

  managed_hermitian_adaptor()
    : m_data(new AdaptedMatrix()), super(*m_data::member)
    {
    }

  managed_hermitian_adaptor(size_type s1, size_type s2)
    : m_data(new AdaptedMatrix(s1, s1)), super(*m_data::member)
    {
      if (s1 != s2)
        throw std::runtime_error( "hermitian matrices are quadratic" );
    }

  managed_hermitian_adaptor(const managed_hermitian_adaptor &m)
    : m_data(new AdaptedMatrix(*m.member)), super(*m_data::member)
    {
    }

  template <class AE>
    managed_hermitian_adaptor(const boost::numeric::ublas::matrix_expression<AE> &ae)
    : m_data(new AdaptedMatrix(ae().size1(), ae().size1())), super(*m_data::member)
    {
      if (ae().size1() != ae().size2())
        throw std::runtime_error( "hermitian matrices are quadratic" );
      super::operator=(ae);
    }

  void resize(size_type s1, size_type s2)
  {
    if (s1 != s2)
      throw std::runtime_error( "hermitian matrices are quadratic" );
    m_data::member->resize(s1, s2);
  }

  void insert(size_type i1, size_type i2, const value_type &val)
  {
    super::operator()(i1, i2) = val;
  }

  void push_back(size_type i1, size_type i2, const value_type &val)
  {
    super::operator()(i1, i2) = val;
  }

  managed_hermitian_adaptor &operator=(const managed_hermitian_adaptor &m)
  {
    *m_data::member.operator=(*m.member);
  }

  template <class AE>
    managed_hermitian_adaptor &operator=(const boost::numeric::ublas::matrix_expression<AE> &ae)
    {
      if (ae().size1() != ae().size2())
        throw std::runtime_error( "hermitian matrices are quadratic" );
      m_data::member->resize(ae.size1(), ae.size2());
      super::operator=(ae);
    }
};




#endif
