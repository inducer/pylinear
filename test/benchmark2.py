import pylinear.matrices as num
import Numeric, random, time
from stopwatch import *

def timeit(f, maxtime = 5):
    start = time.time()
    count = 0
    while time.time() - start <= maxtime:
        f()
        count += 1
    return count / (time.time()-start)

def fillMatrix(matrix):
    rows, columns = matrix.shape
    for row in range(rows):
        for col in range(columns):
            matrix[row,col] = random.normalvariate(0,10)

def bench(module, n):
    mat1 = module.zeros((n,n), module.Float)
    fillMatrix(mat1)
    mat2 = module.zeros((n,n), module.Float)
    fillMatrix(mat2)

    return timeit(lambda : module.matrixmultiply(mat1, mat2))

def run(module, name):
    data_file = file(name, "w")
    sizes = [2,4,8,16,32,64,100,200,300,400]
    for size in sizes:
        this_time = bench(module, size)
        print name, size, this_time
        data_file.write("%f\t%f\n" % (size, this_time))

run(Numeric, ",,times-numeric.data")
run(num, ",,times-pylinear.data")
