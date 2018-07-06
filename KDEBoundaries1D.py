from sklearn.base import BaseEstimator
from scipy.interpolate import interp1d
import numpy as np
from scipy import stats

KERNELS = ['gaussian']
BOUNDARY = ['reflection']


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


class KernelDensityBoundaries1D(BaseEstimator):
    """A Kernel Density Estimator with boundary corrections

    Parameters
    ----------

    kernel : string
        kernel to use. Default: gaussian

    range: list of floats
        List of dimension 2 [Xmin, Xmax] which gives the minimum/maximum value
        of the distribution to estimate

    spline_approx : bool
        if True, the output of score_samples is given by a spline to make
        the calculations faster

    boundary: string
        if None, it calculates the KDE ignoring any boundary condition
        if "reflection", it applies a reflection at both ends of range (see
        http://www.ton.scphys.kyoto-u.ac.jp/~shino/toolbox/reflectedkernel/reflectedkernel.html)

    n_approx : int
        number of points for the spline.


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

        if usespline:
            return self.interpol_(xi)

        elif self.boundary is None:
            if self.kernel == "gaussian":
                KERNEL = GAUSSKERN

            return KERNEL(xi, samplevalues, bw, npts)

        elif self.boundary == "reflection":

            Xmin, Xmax = self.range

            if self.kernel == "gaussian":
                KERNEL = GAUSSKERN

            return KERNEL(xi, samplevalues, bw, npts) + \
                KERNEL(2*Xmin - xi, samplevalues, bw, npts) + \
                KERNEL(2*Xmax - xi, samplevalues, bw, npts)
#                return GAUSSKERN(xi, self.Xvalues_, self.bandwidth,
#                                 self.Xvalues_.shape[0]) + \
#                       GAUSSKERN(2*Xmin - xi, self.Xvalues_, self.bandwidth,
#                                 self.Xvalues_.shape[0]) + \
#                       GAUSSKERN(2*Xmax - xi, self.Xvalues_, self.bandwidth,
#                                 self.Xvalues_.shape[0])

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
        # To make it compatible with
        return(sum(np.log(self.score_samples(X))))
