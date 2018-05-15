import numpy
import matplotlib.pyplot

import algo                         # script with the optimizer functions
from surf import func, grad, hess   # script with the model surface

# Optimize with each algorithm, starting from the following point
x0 = [0.3, 0.6]

x, traj_nr = algo.optimize_newton_raphson(x0=x0, func=func, grad=grad,
                                          hess=hess)

x, traj_qn = algo.optimize_quasi_newton(x0=x0, func=func, grad=grad)

x, traj_gd = algo.optimize_gradient_descent(x0=x0, func=func, grad=grad)

# Generate grid for surface
xmax, ymax = numpy.amax(traj_qn + traj_nr + traj_gd, axis=0)
xmin, ymin = numpy.amin(traj_qn + traj_nr + traj_gd, axis=0)
X = numpy.linspace(xmin, xmax)
Y = numpy.linspace(ymin, ymax)
Z = func(numpy.array(numpy.meshgrid(X, Y)))

# Plot the Newton-Raphson trajectory
matplotlib.pyplot.contour(X, Y, Z, 20)
matplotlib.pyplot.plot(*zip(*traj_nr), color='k')
matplotlib.pyplot.scatter(*zip(*traj_nr), color='k')
s = '{:d} steps'.format(len(traj_nr))
matplotlib.pyplot.text(0.2, 0.1, s=s, fontsize=18, fontweight='bold')
matplotlib.pyplot.savefig('nr')
matplotlib.pyplot.clf()

# Plot the quasi-Newton trajectory
matplotlib.pyplot.contour(X, Y, Z, 20)
matplotlib.pyplot.plot(*zip(*traj_qn), color='k')
matplotlib.pyplot.scatter(*zip(*traj_qn), color='k')
s = '{:d} steps'.format(len(traj_qn))
matplotlib.pyplot.text(0.2, 0.1, s=s, fontsize=18, fontweight='bold')
matplotlib.pyplot.savefig('qn')
matplotlib.pyplot.clf()

# Plot the gradient descent trajectory
matplotlib.pyplot.contour(X, Y, Z, 20)
matplotlib.pyplot.plot(*zip(*traj_gd), color='k')
matplotlib.pyplot.scatter(*zip(*traj_gd), color='k')
s = '{:d} steps'.format(len(traj_gd))
matplotlib.pyplot.text(0.2, 0.1, s=s, fontsize=18, fontweight='bold')
matplotlib.pyplot.savefig('gd')
