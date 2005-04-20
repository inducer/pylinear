import pylinear.array as num
import Numeric
from stopwatch import *

def fillMatrix(matrix):
    for row in range(rows):
        for col in range(columns):
            result[row,col] = random.normalvariate(0,10)
def creation(module, name):
  job = tJob("creation "+name)
  for i in range(1000*10):
    a = module.array([[1.2,2],[3.5,4.2]])
  job.done()

def multiplication(module, name):
  a = module.array([[5.3,6.3],[7.1,8.1]])
  b = module.array([[1,.52],[3,4.5]])
  job = tJob("multiplication "+name)
  for i in range(1000*10):
    c = module.matrixmultiply(a,b)
  job.done()

def elementAccess(module, name):
  a = module.array([[5.3,6.3],[7.1,8.1]])
  b = module.array([[1,.52],[3,4.5]])
  job = tJob("element access "+name)
  for i in range(1000*10):
    c = a[1,0]
  job.done()

def callBoth(function):
  function(num, "pylinear")
  function(Numeric, "numpy")

callBoth(creation)
callBoth(multiplication)
callBoth(elementAccess)
