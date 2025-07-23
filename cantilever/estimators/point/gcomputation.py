import warnings
import numpy as np
import pandas as pd

from cantilever.estimators.point.basics import BridgePointEstimator
from cantilever.estimators.point.efuncs import psi_bridge_point


class BridgeGComputation(BridgePointEstimator):
    r"""Bridged comparison g-computation estimator for the mean or proportion.

    Let :math:`Y^a` be the potential outcome under action :math:`a`, :math:`A \in \{0, 1, 2\}` be the action,
    :math:`S \in \{0,1\}` be the population indicator, :math:`W` be a set of baseline covariates, and :math:`R=1` denote
    the outcome being observed. The parameter of interest is the comparison between :math:`A=2` and :math:`A=0` in the
    :math:`S=1` population, which can be expressed with the single-span and multi-span form:

    .. math::

        \psi_{SS} = E[Y^2 | S=1] - E[Y^0 | S=1] \\
        \psi_{MS} = \left\{ E[Y^2 | S=1] - E[Y^1 | S=1] \right\} + \left\{ E[Y^1 | S=1] - E[Y^0 | S=1] \right\}


    The bridge g-computation estimator computes both parameters (under the corresponding identification assumptions) by
    fitting an outcome model for :math:`E[Y \mid A,W,S,R=1]`. This model is then used to generate predictions under the
    different actions for the target population, :math:`S=1`. The g-computation estimator for :math:`A:=a` is defined as

    .. math::

        \hat{\mu}_{a,s} = \frac{\sum_{i=1}^{n} S_i \times  m_a(W, S=s; \hat{\beta})}{\sum_{i=1}^{n} S_i}

    where :math:`m_a(W,S=s; \beta)` is the predicted value from the outcome model fit using :math:`S=1` given :math:`W`
    and :math:`A:=a`.

    Here, outcome models are fit stratified by :math:`S`. The multi-span expression indicates that :math:`E[Y^1 | S=1]`
    can be separately estimated using the :math:`S=1` and :math:`S=0` data (i.e., :math:`\hat{\mu}_{1,1}` and
    :math:`\hat{\mu}_{1,0}`). Given the identification assumptions are met, a difference of zero between these
    estimators is expected . A non-zero difference is indicative of at least one assumption being
    violated. See the papers in the references for further details.

    The variance for all parameters is estimated using the empirical sandwich variance estimator, which correctly
    incorporates the uncertainty of the nuisance parameter estimates into the variance for the bridge parameters.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame object consisting of all variables of interest. Note all variables, besides the outcome, with
        missing data will have their corresponding rows dropped (i.e., a baseline complete-case analysis is performed).
    outcome : str
        Column label for the outcome variable.
    action : str
        Column label for the exposure variable.
    sample : str
        Column label for the sample or study indicator variable.
    alpha : float, optional
        Alpha level to use for the confidence intervals.

    Examples
    --------
    ...

    References
    ----------
    Shook-Sa BE, Zivich PN, Rosin SP, Edwards JK, Adimora AA, Hudgens MG, Cole SR. (2024). Fusing Trial Data for
    Treatment Comparisons: Single versus Multi-Span Bridging. *Statistics in Medicine*, 43(4):793-815..
    """
    def __init__(self, data, outcome, action, sample, alpha=0.05, verbose=True, decimals=2):
        # initialize the preceding class (allows for more arguments in init than BaseMeanEstimator)
        super().__init__(data=data, outcome=outcome, action=action, sample=sample,
                         alpha=alpha, verbose=verbose, decimals=decimals)

        # Updating specific parameters for g-computation
        self.__estimator_label__ = "Parametric G-computation"

    def action_model(self, model, init=None, bounds=(0, 1)):
        warnings.warn("`BridgeGComputation` does not use the action model. No action model was fit.",
                      UserWarning)

    def sample_model(self, model, init=None, bounds=(0, 1)):
        warnings.warn("`BridgeGComputation` does not use the sample model. No sample model was fit.",
                      UserWarning)

    def missing_model(self, model, init=None, bounds=(0, 1)):
        warnings.warn("`BridgeGComputation` does not use the missing model. No missing model was fit.",
                      UserWarning)

    def diagnostics_weights(self):
        warnings.warn("`BridgeGComputation` does not use weights, so no weight diagnostics are available.",
                      UserWarning)

    def estimate(self, init=None):
        """Estimate the parameters of interest using bridge g-computation.

        Parameters
        ----------
        init : None, list, ndarray, optional
            Optional list of starting values for the root-finding procedure for: single-span, multi-span, diagnostic,
            mean for A=2, mean for A=1 from S=1, mean for A=1 from S=0, mean for A=0. Therefore, the provided array
            must consist of 7 values. If left as ``None`` (the default), starting values for the root-finding procedure
            are automatically generated.

        Returns
        -------
        None
        """
        # Create variables into NumPy arrays for estimation procedure
        y_nan, a, s, m = self._get_variable_arrays_()
        Xa2, Xa1, Xa0 = self._get_updated_design_matrices_()

        # Estimating equations
        def psi(theta):
            return psi_bridge_point(theta,
                                    y=y_nan, a=a, s=s, m=m,
                                    X=self._outcome_design_matrix_, Xa2=Xa2, Xa1=Xa1, Xa0=Xa0,
                                    out_model=self._outcome_model_dist_,
                                    Z=None, V=None, W=None, a_clip=None, s_clip=None, m_clip=None,
                                    include_missing=False)

        # Solving the estimating equations for the parameter of interest
        init = self._generate_inits_(init=init, n_params=7)
        init = init + list(self._outcome_coefs_)
        self.mestimator = self._fit_mestimator_(estimating_functions=psi, init=init)

    def diagnostics(self):
        """

        Returns
        -------

        """
        self.diagnostics_outcome()
        print("")
        self.diagnostic_shared()
