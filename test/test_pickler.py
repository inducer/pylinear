import pylinear.matrices as num
import pickle

a = num.zeros((4,4,), num.Complex, num.SparseExecuteMatrix)
#a = num.zeros((4,4,), num.Complex)
a[0,0] = 4
a[1,0] = 7
a[3,2] = 9
print a.__getstate__()
a_pickled = pickle.dumps(a)
other_a = pickle.loads(a_pickled)
print other_a, a
