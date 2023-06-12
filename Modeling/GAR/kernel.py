import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class Weighted_kernel:
    """
    A class for performing weighted kernel interpolation for estimating conditional quantiles.

    Parameters
    ----------
    cond_quant : dict
        A dictionary containing the observed conditional quantiles and their corresponding input values. The keys represent the input values (theta) and the values represent the conditional quantiles (q). The dictionary should have hashable keys and values that can be converted to NumPy arrays.

    bandwidth : float or None (default: None), optional
        The smoothing parameter (bandwidth) for the weighted kernel interpolation. If provided, the specified value will be used. If set to None, the bandwidth will be estimated automatically based on the data.
    """
    def __init__(self, cond_quant, bandwidth=None):
            theta = np.array(list(cond_quant.keys()))
            q_values = np.array(list(cond_quant.values()))

            # sort the values based on theta
            sorted_indices = np.argsort(theta)
            theta = theta[sorted_indices]
            q_values = q_values[sorted_indices]

            # estimating the smoothing parameter (bandwidth)
            n = len(q_values)
            if bandwidth:
                h = bandwidth
            else:
                q_std = self.quantile_std(theta, q_values)
                IQR = cond_quant[0.75] - cond_quant[0.25]
                h = 1.06*min(q_std, IQR)*(n**(-1/5))

            # initial inputs and weights
            self.theta = theta
            self.q_values = q_values
            self.bandwidth = h
            self.h = h
            self.w_init = np.ones(n)/n

            # adding special constraint for sum(w)=1
            self.cons = {'type':'eq', 'fun': self.const}

    # weighted_kernel_interpolation
    def w_kernel_cdf(self, x, w):
        """
        Computes the weighted kernel estimate of the cumulative distribution function (CDF) for the given input value(s). The interpolation is performed using the observed conditional quantiles and their weights.

        Parameters
        ----------
        x : scalar, array-like
            The input value(s) for which the CDF estimate is computed. If a scalar, a single CDF estimate is returned. If an array-like object, an array of CDF estimates is returned.

        w : array-like
            The weights for the observed conditional quantiles. The length of the weights array should be the same as the number of observed quantiles.
        
        Returns
        -------
        theta_hat : scalar or ndarray
            The estimated quantile(s) corresponding to the input value(s). If a scalar input is given, a single quantile is returned. If an array-like input is given, an array of quantiles is returned.
        """

        q = self.q_values
        h = self.h
        if np.isscalar(x):
            quant_hats = norm.cdf((x-q) / h)
        else:
            quant_hats = norm.cdf((x[:, np.newaxis]-q) / h)
        theta_hat = quant_hats @ w
        return theta_hat

    def w_kernel_pdf(self, x, w):
        """
        Computes the weighted kernel estimate of the probability density function (PDF) for the given input value(s). The interpolation is performed using the observed conditional quantiles and their weights.

        Parameters
        ----------
        x : scalar, array-like
            The input value(s) for which the PDF estimate is computed. If a scalar, a single PDF estimate is returned. If an array-like object, an array of PDF estimates is returned.

        w : array-like
            The weights for the observed conditional quantiles. The length of the weights array should be the same as the number of observed quantiles.
        
        Returns
        -------
        theta_hat : scalar or ndarray
            The estimated PDF value(s) corresponding to the input value(s). If a scalar input is given, a single PDF value is returned. If an array-like input is given, an array of PDF values is returned.
        """
        q = self.q_values
        h = self.h
        if np.isscalar(x):
            quant_hats = norm.pdf((x-q) / h)
        else:
            quant_hats = norm.pdf((x[:, np.newaxis]-q) / h)
        theta_hat = (quant_hats @ w) / h
        return theta_hat
    
    def w_kernel_loss(self, w):
        theta = self.theta
        q = self.q_values
        theta_hat = self.w_kernel_cdf(q, w)
        main_loss = np.sum(np.power(theta - theta_hat, 2))
        total_loss = main_loss
        return total_loss
    
    @staticmethod
    def const(x):
        return x.sum() - 1

    @staticmethod
    def moving_average(a, n=2) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    
    def quantile_std(self, theta, q):
        density = np.diff(theta)/np.diff(q)
        norm_density = density/np.sum(density)
        q_v = self.moving_average(q, 2)
        q_mean = q_v @ norm_density
        q_std = np.sqrt(np.power(q_v - q_mean, 2) @ norm_density)
        return q_std

    def w_kernel_fit(self):
        res = minimize(self.w_kernel_loss, x0=self.w_init, 
                       method='SLSQP', bounds=[(0, None)],
                       constraints=self.cons, 
                       options={'maxiter':100})
        return res.x
