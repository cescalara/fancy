"""
Generate multivariate von Mises Fisher samples.
This solution originally appears here:
http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
Also see:
Sampling from vMF on S^2:
    https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf



KW: Obtained from jasonlaska/spherecluster/spherecluster/util.py:
https://github.com/jasonlaska/spherecluster. This code is under the MIT License.
"""
import numpy as np


def sample_vMF(mu, kappa, num_samples):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \in R^N with concentration kappa.
    """
    dim = len(mu)
    result = np.zeros((num_samples, dim))
    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa, dim)

        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to(mu)

        # compute new point
        result[nn, :] = v * np.sqrt(1. - w**2) + w * mu

    return result


def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa**2 + dim**2) + 2 * kappa)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x**2)

    while True:
        z = np.random.beta(dim / 2., dim / 2.)
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
            return w


def _sample_orthonormal_to(mu):
    """Sample point on sphere orthogonal to mu."""
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)


def sample_sphere(radius, num_samples):
    '''Sample points from a sphere of radius radius uniformly'''
    u = np.random.rand(num_samples)
    v = np.random.rand(num_samples)

    theta = 2. * np.pi * u
    phi = np.arccos(2 * v - 1.)

    result = radius * np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T
    return result

