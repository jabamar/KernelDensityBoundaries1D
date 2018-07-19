"""
Kernel Density Estimation (only 1D)
-------------------------
"""
from sklearn.base import BaseEstimator
from scipy.interpolate import interp1d
import numpy as np
from scipy import stats

KERNELS = ['gaussian', 'tophat', 'epanechnikov', 'expo', 'linear']
BOUNDARY = ['reflection', 'CowlingHall']


def GAUSSKERN(xi, mean, bw, npts):
    """ GAUSSIAN KERNEL

    Parameters
    __________

    xi: float
        x range value

    mean: float or list of floats
        mean values for the gaussian (values of the sample)

    bw: float
        bandwitdh

    npts: integer
        number of points

    """

    return np.divide(np.sum(stats.norm.pdf(xi, loc=mean, scale=bw)), npts)


def TOPHATKERN(xi, val, bw, npts):
    return np.divide((np.abs(xi - val) < bw).sum(), npts*bw*2)


def EPANECHNIKOVKERN(xi, val, bw, npts):
    return np.divide(np.sum(1-(np.power((xi-val)/bw, 2))), npts*bw)


def EXPOKERN(xi, val, bw, npts):
    return np.divide(np.sum(np.exp(-np.abs(xi-val)/bw)), npts*bw*2)


def LINEARKERN(xi, val, bw, npts):
    temparray = np.abs(xi - val)
    temparray = temparray[temparray < bw]
    return np.divide(np.sum(1-(temparray)/bw), npts*bw)


class KernelDensityBoundaries1D(BaseEstimator):
    """A Kernel Density Estimator with boundary corrections

    Parameters
    ----------

    kernel : string
        kernel to use. Default: gaussian

    range: list of floats
        List of dimension 2 [Xmin, Xmax] which gives the minimum/maximum value
        of the distribution to estimate

    bandwith: float
        Smoothing parameter for the kernel

    boundary: string
        if None, it calculates the KDE ignoring any boundary condition
        if "reflection", it applies a reflection at both ends of range (see
        http://www.ton.scphys.kyoto-u.ac.jp/~shino/toolbox/reflectedkernel/reflectedkernel.html).
        If "CowlingHall", The Cowling and Hall method as shown in
        DOI: 10.1103/PhysRevD.97.115047 (original at
        ttps://www.jstor.org/stable/2345893 ) is performed.

    n_approx : int
        number of points for the spline. A spline is used if n_approx >= 2,
        otherwise this is ignored.


    """

    def __init__(self, kernel="gaussian", range=None, bandwidth=1,
                 boundary=None, n_approx=-1):
        """
        Called when initializing the classifier
        """
        self.kernel = kernel
        self.range = range
        self.n_approx = n_approx
        self.bandwidth = bandwidth
        self.boundary = boundary

        if self.kernel not in KERNELS:
            raise RuntimeError("Kernel not valid!")

        if self.boundary is not None and self.boundary not in BOUNDARY:
            raise RuntimeError("Boundary condition not valid!")

        if self.n_approx >= 2 and self.range is None:
            raise RuntimeError("Provide a valid boundary for the spline")

        self.do_spline = True if self.n_approx >= 2 else False

    def fit(self, X, y=None):
        """
        Fit the 1D Kernel Density Estimator
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features=1)
            List of 1-dimensional data points.  Each row
            corresponds to a single data point.
        """

        if X.shape[1] != 1:
            raise RuntimeError("only valid for 1D!")

        self.Xvalues_ = X.copy()

        # Generate Pseudodata points for Cowling-Hall
        if self.boundary == "CowlingHall":

            Xmin, Xmax = self.range
            Npts = int((self.Xvalues_.shape[0])/3)
            sortpts = np.sort(self.Xvalues_.copy())
            self.Xpseudodata_ = 4*Xmin - 6*sortpts[:Npts] + \
                4*sortpts[:2*Npts:2] - sortpts[:3*Npts:3]

        if self.n_approx >= 2:

            Xmin, Xmax = self.range
            if Xmin >= Xmax:
                raise RuntimeError("Xmin must be smaller than Xmax!")

            Xvals = np.linspace(Xmin, Xmax, self.n_approx)
            self.interpol_ = interp1d(Xvals,
                                      [self.eval(xi, False) for xi in Xvals],
                                      kind="cubic")

        return self

    def eval(self, xi, usespline="False"):
        """ Evaluate point by point

        Parameters
        ----------

        xi : float
            point where the density is evaluated

        usespline : bool
            whether the spline is considered or not
        """

        KERNEL = None
        samplevalues = self.Xvalues_
        bw = self.bandwidth
        npts = self.Xvalues_.shape[0]

        # If spline, we return the interpolation and we skip the rest
        if usespline:
            return self.interpol_(xi)

        # Choose kernel
        if self.kernel == "gaussian":
            KERNEL = GAUSSKERN
        elif self.kernel == "tophat":
            KERNEL = TOPHATKERN
        elif self.kernel == "epanechnikov":
            KERNEL = EPANECHNIKOVKERN
        elif self.kernel == "expo":
            KERNEL = EXPOKERN
        elif self.kernel == "linear":
            KERNEL = LINEARKERN

        if self.boundary is None:

            returnval = KERNEL(xi, samplevalues, bw, npts)

        elif self.boundary == "reflection":

            Xmin, Xmax = self.range

            returnval = KERNEL(xi, samplevalues, bw, npts) + \
                KERNEL(2*Xmin - xi, samplevalues, bw, npts) + \
                KERNEL(2*Xmax - xi, samplevalues, bw, npts)

        elif self.boundary == "CowlingHall":

            returnval = KERNEL(xi, samplevalues, bw, npts) + \
                KERNEL(xi, self.Xpseudodata_, bw, npts)

        return returnval if returnval > 1e-40 else 1e-40

    def score_samples(self, X, y=None):
        """Evaluate the density model on the data.
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features=1)
            An array of points to query.  Last dimension must be 1.
        Returns
        -------
        density : ndarray, shape (n_samples,)
            The array of density evaluations.
        """
        return [self.eval(xi, self.do_spline) for xi in X]

    def score(self, X, y=None):
        """Evaluates the total log probability for the array X
        (as done in the sklearn class)

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features=1)
            An array of points to query.  Last dimension must be 1.
        """
        return(sum(np.log(self.score_samples(X))))
