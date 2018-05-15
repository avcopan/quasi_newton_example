import numpy
import warnings


def hessian_update_sr1(x, g, x0, g0, h0):
    dx = numpy.subtract(x, x0)
    dg = numpy.subtract(g, g0)
    h0dx = numpy.dot(h0, dx)
    ddg = dg - h0dx

    # avoid division by zero
    if numpy.linalg.norm(dx) < numpy.finfo(float).eps:
        return h0
    else:
        h = h0 + numpy.outer(ddg, ddg) / numpy.dot(ddg, dx)
        return h


def enforce_max_step_size(dx, smax):
    s = numpy.linalg.norm(dx)
    return dx if s < smax else dx * smax / s


def optimize_quasi_newton(x0, func, grad, smax=0.3, gtol=1e-5, maxiter=50):
    dim = numpy.size(x0)

    x0 = numpy.array(x0)
    g0 = grad(x0)
    h0 = numpy.linalg.norm(g0) / smax * numpy.eye(dim)

    converged = False

    traj = [x0]

    for iteration in range(maxiter):
        dx = - numpy.dot(numpy.linalg.pinv(h0), g0)
        dx = enforce_max_step_size(dx, smax)
        x = x0 + dx
        traj.append(x)

        g = grad(x)
        gmax = numpy.amax(numpy.abs(g))
        converged = gmax < gtol

        if converged:
            break
        else:
            h = hessian_update_sr1(x=x, g=g, x0=x0, g0=g0, h0=h0)
            x0 = x
            g0 = g
            h0 = h

    if not converged:
        warnings.warn("Did not converge!")

    return x, traj


def optimize_newton_raphson(x0, func, grad, hess, smax=0.3, gtol=1e-5,
                            maxiter=50):
    x0 = numpy.array(x0)

    converged = False

    traj = [x0]

    for iteration in range(maxiter):
        g0 = grad(x0)
        h0 = hess(x0)
        dx = - numpy.dot(numpy.linalg.pinv(h0), g0)
        dx = enforce_max_step_size(dx, smax)
        x = x0 + dx
        traj.append(x)

        g = grad(x)
        gmax = numpy.amax(numpy.abs(g))
        converged = gmax < gtol

        if converged:
            break
        else:
            x0 = x

    if not converged:
        warnings.warn("Did not converge!")

    return x, traj


def optimize_gradient_descent(x0, func, grad, smax=0.3, gtol=1e-5, maxiter=50):
    x0 = numpy.array(x0)
    g0 = grad(x0)
    s = smax

    converged = False

    traj = [x0]

    for iteration in range(maxiter):
        dx = - s * g0 / numpy.linalg.norm(g0)
        dx = enforce_max_step_size(dx, smax)
        x = x0 + dx
        traj.append(x)

        g = grad(x)
        gmax = numpy.amax(numpy.abs(g))
        converged = gmax < gtol

        if converged:
            break
        else:
            dg = g - g0
            s = numpy.vdot(dx, dg) * numpy.linalg.norm(g0) / numpy.vdot(dg, dg)
            x0 = x
            g0 = g

    if not converged:
        warnings.warn("Did not converge!")

    return x, traj
