import math
import cmath
import pylinear.matrices as num
#import Numeric as num
#import numarray as num
import stopwatch
import random
from pylinear.matrix_tools import *





def makePermutationMatrix(permutation, typecode):
  size = len(permutation)
  result = num.zeros((size,size), typecode)
  for index, value in zip(range(size), permutation):
    result[index,value] = 1
  return result



  
def _test():
  size = 10
  job = stopwatch.tJob( "make spd" )
  for i in range(100):
    A = makeRandomSPDMatrix(size, num.Complex64)
  job.done()
  #print A




if __name__ == "__main__":
  _test()
