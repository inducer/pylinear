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




#include "array.hpp"




namespace
{
  template <typename ValueType>
  static void exposeAll(ValueType, const std::string &python_eltypename)
  {
    exposeVectorType(ublas::vector<ValueType>(), "Vector", python_eltypename);
  }




  /*
  template <class Vec>
  class multidim_view
  {
    private:
      boost::shared_ptr<Vec>    m_vector;
      std::vector<unsigned>     m_dimensions;
      unsigned                  m_offset;
      std::vector<unsigned>     m_strides;

    public:
      multidim_view(boost::shared_ptr<Vec> &vector, 
          const std::vector<unsigned> &dimensions)
        : m_vector(vector), m_dimensions(dimensions)
      { }

      multidim_view(multidim_view &src)
        : m_vector(src.m_vector), m_dimensions(src.m_dimensions)
      { }



      Vec &vector() const
      { return m_vector; }
  };
  */
}



void pylinear_expose_vector()
{
  EXPOSE_ALL_TYPES;
}

