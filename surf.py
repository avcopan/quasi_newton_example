import numpy


def func(z):
    x, y = z
    return (1. - y ** 2) * x ** 2 * numpy.exp(-x ** 2) + 1 / 2. * y ** 2


def grad(z):
    gx = dfdx(z)
    gy = dfdy(z)
    return numpy.array([gx, gy])


def hess(z):
    hxx = d2fdx2(z)
    hxy = d2fdxdy(z)
    hyy = d2fdy2(z)
    return numpy.array([[hxx, hxy], [hxy, hyy]])


def dfdx(z):
    x, y = z
    return 2 * (1. - y ** 2) * x * (1. - x ** 2) * numpy.exp(-x ** 2)


def dfdy(z):
    x, y = z
    return y * (1. - 2 * x ** 2 * numpy.exp(-x ** 2))


def d2fdx2(z):
    x, y = z
    return (2 * (1. - y ** 2) * (1. - 5. * x ** 2 + 2. * x ** 4) *
            numpy.exp(-x ** 2))


def d2fdxdy(z):
    x, y = z
    return -4 * y * x * (1. - x ** 2) * numpy.exp(-x ** 2)


def d2fdy2(z):
    x, y = z
    return 1. - 2 * x ** 2 * numpy.exp(-x ** 2)
