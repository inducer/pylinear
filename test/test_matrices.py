import pylinear.matrices as num

mat = num.zeros((3, 3), num.Complex64)
mat[1,2] += 5+3j
mat[2,1] += 7-8j
#print num.hermite(mat)[1]
#print num.hermite(mat)[1,:]

vec = num.zeros((3,), num.Complex64)
for i in vec.indices():
  vec[i] = 17
mat[0] = vec

for i in mat:
  print i
for i in mat.indices():
  print i, mat[i]

print sum(vec)
print num.matrixmultiply(mat, mat)
