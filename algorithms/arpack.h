#ifndef HEADER_SEEN_ARPACK_H
#define HEADER_SEEN_ARPACK_H




#include <matrix_operator.h>
#include <arpack_proto.h>
#include <helpers.h>




namespace arpack
{
  enum which_eigenvalues { LARGEST_MAGNITUDE,
    SMALLEST_MAGNITUDE,
    LARGEST_REAL_PART,
    SMALLEST_REAL_PART,
    LARGEST_IMAGINARY_PART,
    SMALLEST_IMAGINARY_PART
  };

  namespace detail
  {
    const char *mapWhichToString(which_eigenvalues we)
    {
      switch (we)
      {
        case LARGEST_MAGNITUDE: return "LM";
        case SMALLEST_MAGNITUDE: return "SM";
        case LARGEST_REAL_PART: return "LR";
        case SMALLEST_REAL_PART: return "SR";
        case LARGEST_IMAGINARY_PART: return "LI";
        case SMALLEST_IMAGINARY_PART: return "SI";
      }
    }
  }

  enum arpack_mode { 
    REGULAR_NON_GENERALIZED = 1,
    REGULAR_GENERALIZED = 2,
    SHIFT_AND_INVERT_GENERALIZED = 3
  };

  template <typename ValueType>
  struct results
  {
    std::vector<std::complex<ValueType> > m_eigenvalues;
    std::vector<ublas::vector<ValueType> > > m_eigenvectors;
  };




  namespace detail
  {
    template <typename ValueType>
    void doReverseCommunication(
        matrix_operator<ValueType> &op, 
        matrix_operator <ValueType> &m,
        arpack_mode mode,
        ValueType spectral_shift,
        results<ValueType> &results,
        int number_of_eigenvalues,
        which_eigenvalues which_e = LARGEST_MAGNITUDE,
        ValueType tolerance = 1e-8,
        bool m_is_identity = false,
        int max_iterations = 0
        )
    {
      typedef 
        typename helpers::decomplexify<ValueType>::type
        base_type;

      int ido = 0;
      char bmat = m_is_identity ? 'I' : 'G';
      int n = a.size1();

      if (n != a.size2() || n != m.size1() || n != m.size2())
        throw std::runtime_error("arpack: matrix sizes don't match.");

      const char *which = mapWhichToString(which_e);

      ValueType residual[n];

      int ncv = number_of_eigenvalues * 3;
      ValueType v[ncv * n];
      int ldv = ncv;

      int iparam[11];
      iparam[1-1] = 1; // exact shifts
      iparam[2-1] = 0; // unused
      iparam[3-1] = max_iterations != 0 ? max_iterations : 10000 * n;
      iparam[4-1] = 1; // block size
      iparam[5-1] = 0; // NCONV
      iparam[6-1] = 0; // IUPD, unused
      iparam[7-1] = mode;
      iparam[8-1] = 0; // NP, something to do with user-provided shifts
      iparam[9-1] = 0; // NUMOP
      iparam[10-1] = 0; // NUMOPB
      iparam[11-1] = 0; // NUMREO

      int info = 0; // we're not specifying a previous residual

      int ipntr[14];

      ValueType workd[3*n];
      int lworkl = 3 * ncv * ncv + 5 * ncv;
      ValueType workl[lworkl];
      double rwork[ncv];

      do
      {
        naupd(
            &ido,
            &bmat,
            &n,
            which,
            &number_of_eigenvalues,
            &tolerance,
            residual,
            &ncv,
            v,
            &ldv,
            iparam,
            ipntr,
            workd,
            workl,
            &lworkl,
            rwork,
            &info
            );

        switch (info)
        {
          case 0:
            break;
          case 1:
            throw runtime_error("arpack: performed max. number of iterations (1)");
          case 3:
            throw runtime_error("arpack: could not apply shifts (3)");
          case -1:
            throw runtime_error("arpack: n not positive (-1)");
          case -2:
            throw runtime_error("arpack: nev not positive (-2)");
          case -3:
            throw runtime_error("arpack: ncv-nev >= 2 and <= n (-3)");
          case -4:
            throw runtime_error("arpack: max_iterations must be bigger than zero (-4)");
          case -5:
            throw runtime_error("arpack: invalid WHICH (-5)");
          case -6:
            throw runtime_error("arpack: invalid BMAT (-6)");
          case -7:
            throw runtime_error("arpack: work array too short (-7)");
          case -8:
            throw runtime_error("arpack: LAPACK error (-8)");
          case -9:
            throw runtime_error("arpack: starting vector is zero (-9)");
          case -10:
            throw runtime_error("arpack: invalid MODE (-10)");
          case -11:
            throw runtime_error("arpack: MODE and BMAT don't agree (-11)");
          case -12:
            throw runtime_error("arpack: ISHIFT invalid (-12)");
          default:
            throw runtime_error("arpack: invalid naupd error code");
        }

        if (ido == -1 || ido == 1 || ido == 2)
        {
          vector<ValueType> operand, result;
          operand.resize(n);
          result.resize(n);

          ValueType *x = workd + inptr[1-1] - 1;
          ValueType *y = workd + inptr[2-1] - 1;

          for (unsigned i = 0; i < n; i++)
            operand[i] = x[i];

          if (ido == 2)
            m.apply(operand, result);
          else
            op.apply(operand, result);

          for (unsigned i = 0; i < n; i++)
            y[i] = result[i];
        }
        else if (ido == 99) /*nothing*/;
        else
          throw std::runtime_error("arpack: reverse communication failure");
      }
      while (ido != 99);

      if (max_iterations != 0)
      {
        if (max_iterations <= iparam[2])
          throw std::runtime_error("arpack: hit iteration count limit");
        max_iterations -= iparam[2];
      }

      {
        // prepare for call to neupd
        int rvec = 1;
        char howmny = 'A';
        int select[ncv]; // no-op
        ValueType d[nev+1];
        ValueType z[nev*n];
        int ldz = nev;
        std::complex<base_type> sigma;
        ValueType workev[2*ncv];

        zneupd(
            &rvec, &howmny, select, d, z, &ldz, &sigma, workev,
            // naupd parameters follow
            &bmat,
            &n,
            which,
            &number_of_eigenvalues,
            &tolerance,
            residual,
            &ncv,
            v,
            &ldv,
            iparam,
            ipntr,
            workd,
            workl,
            &lworkl,
            rwork,
            &info
            );
        switch (info)
        {
          case 0:
            break;
          case 1:
            throw runtime_error("arpack: schur form could not be reordered (1)");
          case -1:
            throw runtime_error("arpack: n not positive (-1)");
          case -2:
            throw runtime_error("arpack: nev not positive (-2)");
          case -3:
            throw runtime_error("arpack: ncv-nev >= 2 and <= n (-3)");
          case -5:
            throw runtime_error("arpack: invalid WHICH (-5)");
          case -6:
            throw runtime_error("arpack: invalid BMAT (-6)");
          case -7:
            throw runtime_error("arpack: work array too short (-7)");
          case -8:
            throw runtime_error("arpack: LAPACK error (-8)");
          case -9:
            throw runtime_error("arpack: LAPACK _trevc failed (-9)");
          case -10:
            throw runtime_error("arpack: invalid MODE (-10)");
          case -11:
            throw runtime_error("arpack: MODE and BMAT don't agree (-11)");
          case -12:
            throw runtime_error("arpack: HOWNY = S invalid (-12)");
          case -13:
            throw runtime_error("arpack: HOWNY and RVEC don't agree (-13)");
          case -14:
            throw runtime_error("arpack: no eigenvalues found (-14)");
          default:
            throw runtime_error("arpack: invalid neupd error code");
        }
      }
    }
  }
}



#endif
