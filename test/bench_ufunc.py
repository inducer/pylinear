from stopwatch import *
from test_tools import *
import pylinear.matrices as num

mat = makeFullRandomMatrix(1000, num.Float)
job = tJob("ufunc")
for i in range(10):
  mat = num.cos(mat)
job.done()
#print mat
