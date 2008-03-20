from pytools import Table

def fill_matrix(matrix):
      for row in range(rows):
          for col in range(columns):
              result[row,col] = random.normalvariate(0,10)

tbl = Table()
tbl.add_row(("task", "module", "size", "time/el"))

class Job:
    def __init__(self, name, module, size):
        self.name = name
        self.module = module.__name__
        self.size = size
        from time import time
        self.start = time()

    def done(self):
        from time import time
        end = time()

        if size:
            dur = (end-self.start) / self.size
        else:
            dur = end-self.start

        tbl.add_row((self.name, self.module, str(self.size), dur))

def creation(module):
    job = Job("creation", module, None)
    for i in range(1000*10):
      a = module.array([[1.2,2],[3.5,4.2]])
    job.done()

def multiplication(module):
    a = module.array([[5.3,6.3],[7.1,8.1]])
    b = module.array([[1,.52],[3,4.5]])
    job = Job("matrix mult", module, None)
    for i in range(1000*10):
      c = module.matrixmultiply(a,b)
    job.done()

def addition(module, size):
    a = module.array(range(size))
    b = module.array(range(size)[::-1])
    job = Job("addition", module, size)
    for i in range(1000*10):
      c = a+b
      c = a+b
      c = a+b
      c = a+b
    job.done()

def broadcast_addition(module, size):
    a = module.array(range(size))
    job = Job("broadcast_addition", module, size)
    for i in range(1000*10):
      c = a+0
      c = a+0
      c = a+0
      c = a+0
    job.done()

def element_access(module):
    a = module.array([[5.3,6.3],[7.1,8.1]])
    job = Job("element access size:%d module:%s " % (0, module.__name__))
    for i in range(1000*10):
      c = a[1,0]
    job.done()

def call_with_both(function, *args):
    import pylinear.array
    import numpy
    function(pylinear.array, *args)
    function(numpy, *args)

#call_with_both(creation)
#call_with_both(multiplication)
#call_with_both(element_access)
for size in [3, 30, 300, 3000]:
    call_with_both(addition, size)
    call_with_both(broadcast_addition, size)

print tbl
