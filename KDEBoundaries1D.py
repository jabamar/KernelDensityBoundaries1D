from sklearn.base import BaseEstimator
from scipy.interpolate import interp1d
import numpy as np
from scipy import stats

KERNELS = ['gaussian']


class KernelDensityBoundaries1D(BaseEstimator):
    """A Kernel Density Estimator with boundary corrections

    Parameters
    ----------

    kernel : string
        kernel to use. Default: gaussian

    boundaries: list of floats
        List of dimension 2 [Xmin, Xmax] which gives the minimum/maximum value
        of the distribution to estimate

    spline_approx : bool
        if True, the output of score_samples is given by a spline to make
        the calculations faster

    n_approx : int
        number of points for the spline.


    """

    def __init__(self, kernel="gaussian", boundaries=None, bandwidth=1,
                 spline_approx=False, n_approx=-1):
        """
        Called when initializing the classifier
        """
        self.kernel = kernel
        self.boundaries = boundaries
        self.n_approx = n_approx
        self.bandwidth = bandwidth

        if self.kernel not in KERNELS:
            raise RuntimeError("Kernel not valid!")

        if self.n_approx >= 2 and self.boundaries is None:
            raise RuntimeError("Provide a valid boundary for the spline")

        self.do_spline = True if self.n_approx >= 2 else False

        print("Hola")

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

        if self.boundaries is not None:

            Xmin, Xmax = self.boundaries
            if Xmin >= Xmax:
                raise RuntimeError("Xmin must be smaller than Xmax!")

            Xvals = np.linspace(Xmin, Xmax, self.n_approx)
            self.interpol_ = interp1d(Xvals, self._eval(Xvals, False),
                                      kind="cubic")

        return self

    def _eval(self, xi, usespline="False"):
        """ Evaluate point by point

        Parameters
        ----------

        xi : float
            point where the density is evaluated

        usespline : bool
            whether the spline is considered or not
        """

        if usespline:
            return self.interpol_(xi)
        else:
            if self.kernel == "gaussian":
                return sum(stats.norm.pdf(xi, loc=xcent, scale=self.bandwidth)
                           for xcent in self.Xvalues_)/self.Xvalues_.shape[0]

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
        return [self._eval(xi, self.do_spline) for xi in X]

    def score(self, X, y=None):
        # To make it compatible with
        return(sum(np.log(self.score_samples(X))))
