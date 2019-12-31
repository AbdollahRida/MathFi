import warnings

import numpy as np

from tqdm import tqdm


class FBM(object):

    def __init__(self, n, hurst, length=1, method="hosking"):
        """
        Instantiate the FBM.
        Methods available are Hosking and Cholesky.   
        """
        self._methods = {"cholesky": self._cholesky, "hosking": self._hosking}
        self.n = n
        self.hurst = hurst
        self.length = length
        self.method = method
        self._fgn = self._methods[self.method]
        # Some reusable values to speed up Monte Carlo.
        self._cov = None
        self._eigenvals = None
        self._C = None
        # Flag if some params get changed
        self._changed = False

    def __str__(self):
        """
        Str method.
        """
        return (
            "fBm ("
            + str(self.method)
            + ") on [0, "
            + str(self.length)
            + "] with Hurst value "
            + str(self.hurst)
            + " and "
            + str(self.n)
            + " increments"
        )

    def __repr__(self):
        """
        Repr method.
        """
        return (
            "FBM(n="
            + str(self.n)
            + ", hurst="
            + str(self.hurst)
            + ", length="
            + str(self.length)
            + ', method="'
            + str(self.method)
            + '")'
        )

    @property
    def n(self):
        """
        Get the number of increments.
        """
        return self._n

    @n.setter
    def n(self, value):
        if not isinstance(value, int) or value <= 0:
            raise TypeError("Number of increments must be a positive int.")
        self._n = value
        self._changed = True

    @property
    def hurst(self):
        """
        Hurst parameter.
        Defines the roughness of the fBm.
        """
        return self._hurst

    @hurst.setter
    def hurst(self, value):
        if not isinstance(value, float) or value <= 0 or value >= 1:
            raise ValueError("Hurst parameter must be in interval (0, 1).")
        self._hurst = value
        self._changed = True

    @property
    def length(self):
        """
        Get the length of process.
        """
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Length of fbm must be greater than 0.")
        self._length = value
        self._changed = True

    @property
    def method(self):
        """
        Get the algorithm used to generate.
        """
        return self._method

    @method.setter
    def method(self, value):
        if value not in self._methods:
            raise ValueError("Method must be 'hosking' or 'cholesky'.")
        self._method = value
        self._fgn = self._methods[self.method]
        self._changed = True

    def fbm(self):
        """
        Sample the fractional Brownian motion.
        """
        return np.insert(self.fgn().cumsum(), [0], 0)

    def fgn(self):
        """
        Sample the fractional Gaussian noise.
        We rescale using the self-similarity property.
        """
        scale = (1.0 * self.length / self.n) ** self.hurst
        gn = np.random.normal(0.0, 1.0, self.n)

        # If hurst == 1/2 then just return Gaussian noise
        if self.hurst == 0.5:
            return gn * scale
        else:
            fgn = self._fgn(gn)

        # Scale to interval [0, L]
        return fgn * scale

    def times(self):
        """
        Get times associated with the fbm/fgn samples.
        """
        return np.linspace(0, self.length, self.n + 1)

    def _autocovariance(self, k):
        """
        Autocovariance for fgn.
        """
        return 0.5 * (abs(k - 1) ** (2 * self.hurst) - 2 * abs(k) ** (2 * self.hurst) + abs(k + 1) ** (2 * self.hurst))

    def _cholesky(self, gn):
        """
        Generate a fgn realization using the Cholesky method.
        """
        # Monte carlo consideration
        if self._C is None or self._changed:
            # Generate covariance matrix
            G = np.zeros([self.n, self.n])
            for i in tqdm(range(self.n)):
                for j in tqdm(range(i + 1)):
                    G[i, j] = self._autocovariance(i - j)

            # Cholesky decomposition
            self._C = np.linalg.cholesky(G)
            self._changed = False

        # Generate fgn
        fgn = np.dot(self._C, np.array(gn).transpose())
        fgn = np.squeeze(fgn)
        return fgn

    def _hosking(self, gn):
        """
        Generate a fGn realization using Hosking's method.
        """
        fgn = np.zeros(self.n)
        phi = np.zeros(self.n)
        psi = np.zeros(self.n)
        # Monte carlo consideration
        if self._cov is None or self._changed:
            self._cov = np.array([self._autocovariance(i) for i in range(self.n)])
            self._changed = False

        # First increment from stationary distribution
        fgn[0] = gn[0]
        v = 1
        phi[0] = 0

        # Generate fgn realization with n increments of size 1
        for i in tqdm(range(1, self.n)):
            phi[i - 1] = self._cov[i]
            for j in tqdm(range(i - 1)):
                psi[j] = phi[j]
                phi[i - 1] -= psi[j] * self._cov[i - j - 1]
            phi[i - 1] /= v
            for j in tqdm(range(i - 1)):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
            v *= 1 - phi[i - 1] * phi[i - 1]
            for j in tqdm(range(i)):
                fgn[i] += phi[j] * fgn[i - j - 1]
            fgn[i] += np.sqrt(v) * gn[i]

        return fgn


def fbm(n, hurst, length=1, method="hosking"):
    """
    One off sample of fBm.
    """
    f = FBM(n, hurst, length, method)
    return f.fbm()


def fgn(n, hurst, length=1, method="hosking"):
    """
    One off sample of fGn.
    """
    f = FBM(n, hurst, length, method)
    return f.fgn()


def times(n, length=1):
    """
    Generate the times associated with increments n and length.
    """
    return np.linspace(0, length, n + 1)
