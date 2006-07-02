import pylinear.array as num
import pylinear.toybox as toybox

def f(t, y):
    return num.array([y[1], -y[0]])

timesteps = toybox.integrate_ode(num.array([0,1]), f, 0, 10)

outf = file(",ode.dat", "w")

for t, v in timesteps:
    outf.write("%f\t%f\n" % (t,v[0]))
