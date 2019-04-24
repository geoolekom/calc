import numpy
from matplotlib import pyplot


def n(v, xi, phi):
    beta_rev = numpy.cos(phi) * numpy.cos(phi)
    a = - xi * xi / 3
    b = 2 * xi * xi * numpy.tan(phi) * numpy.tan(phi) / 3
    zero_n = 1 + a * beta_rev + 2 * b * beta_rev * beta_rev
    if v is None:
        return 0
    else:
        z = v * v / 2
        return numpy.exp(- z / beta_rev) * (zero_n - z * (a + 2 * b * beta_rev) + b * z * z) * beta_rev


xi = 0.1
phi = numpy.linspace(- 1.6, 1.6, 50)
f = n(0, xi, phi) - n(0.25, xi, phi)

pyplot.plot(phi, f)
pyplot.show()
