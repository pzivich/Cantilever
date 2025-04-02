import warnings
import numpy as np
import pandas as pd
from delicatessen.utilities import inverse_logit

from cantilever.formulas import get_design_matrix
from cantilever.estimators.efuncs import psi_bridge_point
from cantilever.estimators.utils import fit_mestimator, print_nuisance_model_results
from cantilever.estimators.point.basics import BridgePointEstimator, psi_action, psi_sample, psi_missing


class BridgeIPW(BridgePointEstimator):
    """Bridged comparison inverse probability weighting (IPW) estimator for point outcome data.

    Here, the Hajek estimator is used, rather than the Horvitz-Thompson.

    """
    def __init__(self, data, outcome, action, sample, alpha=0.05, verbose=True, decimals=2):
        # initialize the preceding class (this allows for more arguments in init than available in BaseMeanEstimator)
        super().__init__(data=data, outcome=outcome, action=action, sample=sample, alpha=alpha,
                         verbose=verbose, decimals=decimals)
        self.__estimator_label__ = "Inverse Probability Weighting - Hajek"
        self.outcome_model = None

    def outcome_model(self, model, model_type='linear', init=None):
        warnings.warn("`BridgeIPW` does not use the outcome model. No outcome model was fit.",
                      UserWarning)

    def estimate(self, init=None):
        """

        Parameters
        ----------
        init

        Returns
        -------

        """
        y_nan, a, s, m = self._get_variable_arrays_()
        include_m = self._missing_nuisance_model_ is not None
        if self._missing_nuisance_model_ is None:
            nuisance_inits = list(self._action_coefs_) + list(self._sample_coefs_)
        else:
            nuisance_inits = list(self._action_coefs_) + list(self._sample_coefs_) + list(self._missing_coefs_)

        # Defining the estimating functions
        def psi(theta):
            return psi_bridge_point(theta,
                                    y=y_nan, a=a, s=s, m=m,
                                    X=None, Xa2=None, Xa1=None, Xa0=None, out_model=None,
                                    Z=self._action_design_matrix_, V=self._sample_design_matrix_,
                                    W=self._missing_design_matrix_, a_clip=self._truncation_pract_,
                                    s_clip=self._truncation_prsamp_, m_clip=self._truncation_prmiss_,
                                    include_missing=include_m,
                                    aipw_implementation=None)

        # Initial values for the root-finding procedure (iteratively builds on previous fits)
        init = self._generate_inits_(init=init, n_params=7)
        init = init + nuisance_inits

        # Solving the estimating equations for the parameter of interest
        self.mestimator = self._fit_mestimator_(estimating_functions=psi, init=init)
