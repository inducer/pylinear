# Python implementation of bicgstab, dumped.

def bicgstab(A, b, tolb = 1e-6):
    x = num.zeros((A.shape[1],), A.typecode())
    r = b - num.matrixmultiply(A, x)
    normr = mtools.norm2(r)

    rt = r;                            
    normrmin = normr;                  
    rho = 1;
    omega = 1;
    stag = 0;                          
    alpha = 0
   
    for i in range(0, 1000):
        rho1 = rho
        rho = mtools.sp(r, rt)
        if rho == 0:
            raise RuntimeError, "rho"
        if i == 0:
            p = r
        else:
            beta = (rho/rho1)*(alpha/omega)
            if beta == 0:
                raise RuntimeError, "beta"
            p = r + beta * (p - omega * v)

        ph = p
        v = num.matrixmultiply(A, ph);
        rtv = mtools.sp(v, rt)
        if rtv == 0:
            raise RuntimeError
        alpha = rho / rtv;
        if alpha == 0:
            raise RuntimeError, "alpha"

        xhalf = x + alpha * ph;        
        rhalf = b - num.matrixmultiply(A, xhalf);         
        normr = mtools.norm2(rhalf);

        if normr <= tolb:
            return xhalf

        if normr < normrmin:
            normrmin = normr;
            xmin = xhalf;
            imin = i - 0.5;

        s = r - alpha * v
        sh = s
        t = num.matrixmultiply(A, sh)
        tt = mtools.norm2squared(t)
        if tt == 0:
            raise RuntimeError, "tt"
        omega = mtools.sp(s, t)/ tt;
        if omega == 0:  
            raise RuntimeError, "omega"

        x = xhalf + omega * sh;     
        normr = mtools.norm2(b - num.matrixmultiply(A, x));
        print normr

        if normr <= tolb:
            return x

        if normr < normrmin:     
            normrmin = normr
            xmin = x
            imin = i

        r = s - omega * t

    raise RuntimeError, "maxit"

