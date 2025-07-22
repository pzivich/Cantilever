import warnings
import numpy as np
import pandas as pd

from cantilever.estimators.point.basics import BridgePointEstimator
from cantilever.estimators.point.efuncs import psi_weighted_outcome, psi_bridge_point
from cantilever.estimators.utils import print_nuisance_model_results


class BridgeAIPW(BridgePointEstimator):
    """Bridged comparison augmented inverse probability weighting (AIPW) estimator for point outcome data.

    Here, weighted regression AIPW is used.

    """

    def __init__(self, data, outcome, action, sample, alpha=0.05, verbose=True, decimals=2):
        # initialize the preceding class (this allows for more arguments in init than available in BaseMeanEstimator)
        super().__init__(data=data, outcome=outcome, action=action, sample=sample, alpha=alpha,
                         verbose=verbose, decimals=decimals)
        self.__estimator_label__ = "Augmented Inverse Probability Weighting"

    def estimate(self, init=None):
        """

        Parameters
        ----------
        init

        Returns
        -------

        """
        y_nan, a, s, m = self._get_variable_arrays_()
        Xa2, Xa1, Xa0 = self._get_updated_design_matrices_()
        include_m = self._missing_nuisance_model_ is not None

        # Nuisance parameters re-estimated for wr-AIPW
        nuisance_inits = self._weighted_outcome_model_()

        # Defining the estimating functions
        def psi(theta):
            return psi_bridge_point(theta,
                                    y=y_nan, a=a, s=s, m=m,
                                    X=self._outcome_design_matrix_, Xa2=Xa2, Xa1=Xa1, Xa0=Xa0,
                                    out_model=self._outcome_model_dist_,
                                    Z=self._action_design_matrix_, V=self._sample_design_matrix_,
                                    W=self._missing_design_matrix_, a_clip=self._truncation_pract_,
                                    s_clip=self._truncation_prsamp_, m_clip=self._truncation_prmiss_,
                                    include_missing=include_m)

        # Initial values for the root-finding procedure (iteratively builds on previous fits)
        init = self._generate_inits_(init=init, n_params=7)
        init = init + nuisance_inits

        # Solving the estimating equations for the parameter of interest
        self.mestimator = self._fit_mestimator_(estimating_functions=psi, init=init)

    def _weighted_outcome_model_(self):
        y_nan, a, s, m = self._get_variable_arrays_()
        include_m = self._missing_nuisance_model_ is not None
        if self._missing_nuisance_model_ is None:
            nuisance_inits = list(self._outcome_coefs_) + list(self._action_coefs_) + list(self._sample_coefs_)
            missing_labels = []
        else:
            nuisance_inits = (list(self._outcome_coefs_) + list(self._action_coefs_)
                              + list(self._sample_coefs_) + list(self._missing_coefs_))
            missing_labels = self._missing_coefs_labels_

        def psi(theta):
            return psi_weighted_outcome(theta,
                                        y=y_nan, a=a, s=s, m=m,
                                        X=self._outcome_design_matrix_, model=self._outcome_model_dist_,
                                        Z=self._action_design_matrix_, V=self._sample_design_matrix_,
                                        W=self._missing_design_matrix_, a_clip=self._truncation_pract_,
                                        s_clip=self._truncation_prsamp_, m_clip=self._truncation_prmiss_,
                                        include_missing=include_m)

        # Solving the estimating equations for weighted model
        estr = self._fit_mestimator_(estimating_functions=psi, init=nuisance_inits)
        if self._verbose_:
            print("====================================================================")
            print("Weighted Outcome Nuisance Model")
            print("--------------------------------------------------------------------")
            self._print_nuisance_fit_details_(n_obs=self.__outcomes_n__,
                                              dep_var=self.outcome,
                                              family=self._outcome_model_dist_)
            print("====================================================================")
            print_nuisance_model_results(labels=(["S=1 : " + l for l in self._outcome_coefs_labels_]
                                                 + ["S=0 : " + l for l in self._outcome_coefs_labels_]
                                                 + ["S=1 : " + l for l in self._action_coefs_labels_]
                                                 + ["S=0 : " + l for l in self._action_coefs_labels_]
                                                 + self._sample_coefs_labels_
                                                 + missing_labels),
                                         m_estimator=estr, decimals=self._decimals_,
                                         subset=[i for i in range(len(self._outcome_coefs_labels_)*2)])
            print("====================================================================")

        return list(estr.theta)
