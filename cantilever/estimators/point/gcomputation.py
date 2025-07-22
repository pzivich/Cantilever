import warnings
import numpy as np
import pandas as pd

from cantilever.estimators.point.basics import BridgePointEstimator
from cantilever.estimators.point.efuncs import psi_bridge_point


class BridgeGComputation(BridgePointEstimator):
    """Bridged comparison g-computation estimator for point outcome data.

    Let :math:`Y^a` be the potential outcome under action :math:`a`, :math:`A \in \{0, 1, 2\}` be the action, and
    :math:`S \in \{0,1\}` be the population indicator. Bridge g-computation operates by fitting outcomes models
    stratified by :math:`S`. The outcome models are then used to generate predictions under the non-overlapping actions
    between trials for the target population. Both the single-span and multi-span forms of the bridged comparison are
    computed:

    .. math::

        \psi_{SS} = E[Y^2 | S=1] - E[Y^0 | S=1] \\
        \psi_{MS} = \left\{ E[Y^2 | S=1] - E[Y^1 | S=1] \right\} + \left\{ E[Y^1 | S=1] - E[Y^0 | S=1] \right\}

    By stratifying outcome models by the study, a diagnostic procedure is enabled. Namely, one can compare the
    predicted means for the shared action between studies from models fit to each study. The form of the diagnostic is

    .. math::

        \psi_{D} = E[Y^1 | S=1] - E[Y^1 | S=1]

    where the first expectation is estimated using the outcome model fit using :math:`S=1` data, and the second is
    estimated using the outcome model fit using :math:`S=0` data. A non-zero difference between the predicted means is
    indicative of an assumption being violated for the fusion.

    Note
    ----
    The variance is estimated using the empirical sandwich variance estimator (implemented via ``delicatessen``), which
    incorporates the uncertainty of the nuisance parameter estimates correctly into the variance for the bridge
    comparisons.


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

    References
    ----------
    Shook-Sa BE, Zivich PN, Rosin SP, Edwards JK, Adimora AA, Hudgens MG, Cole SR. (2023). Fusing Trial Data for
    Treatment Comparisons: Single versus Multi-Span Bridging. *arXiv:2305.00845*.
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
        """Estimate the bridged comparison.

        Parameters
        ----------
        init

        Returns
        -------

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
