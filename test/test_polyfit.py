import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools
import math




def f(x):
  return math.sqrt(math.fabs(math.sin(5*x)+2*math.cos(x)-1))

x = num.array(mtools.linspace(0, 1))
approximant = mtools.get_approximant(x, f, 10)

x2 = mtools.linspace(0., 1.)
gnuplot_file = file(",,approx.data", "w")
for i in x2:
  gnuplot_file.write("%f %f\n" % (i, approximant(i)))

gnuplot_file = file(",,f.data", "w")
for i in x2:
  gnuplot_file.write("%f %f\n" % (i, f(i)))

