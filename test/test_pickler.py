import pylinear.matrices as num
import pickle

a = num.array([[1,2],[3,4]], num.Complex)
a_pickled = pickle.dumps(a)
other_a = pickle.loads(a_pickled)
print other_a, a
