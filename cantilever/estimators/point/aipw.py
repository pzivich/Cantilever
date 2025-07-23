import warnings
import numpy as np
import pandas as pd

from cantilever.estimators.point.basics import BridgePointEstimator
from cantilever.estimators.point.efuncs import psi_weighted_outcome, psi_bridge_point
from cantilever.estimators.utils import print_nuisance_model_results


class BridgeAIPW(BridgePointEstimator):
    r"""Bridged comparison augmented inverse probability weighting (AIPW) estimator  for the mean or proportion.

    Let :math:`Y^a` be the potential outcome under action :math:`a`, :math:`A \in \{0, 1, 2\}` be the action,
    :math:`S \in \{0,1\}` be the population indicator, :math:`W` be a set of baseline covariates, and :math:`R=1` denote
    the outcome being observed. The parameter of interest is the comparison between :math:`A=2` and :math:`A=0` in the
    :math:`S=1` population, which can be expressed with the single-span and multi-span form:

    .. math::

        \psi_{SS} = E[Y^2 | S=1] - E[Y^0 | S=1] \\
        \psi_{MS} = \left\{ E[Y^2 | S=1] - E[Y^1 | S=1] \right\} + \left\{ E[Y^1 | S=1] - E[Y^0 | S=1] \right\}


    The bridge AIPW estimator computed both parameters (under the corresponding identification assumptions) using the
    weighted-regression implementation of AIPW. First, inverse probability weights are estimated from models for the
    action (:math:`\Pr(A=a \mid W,S=s)`), sampling (:math:`\Pr(S=1 \mid W)`), and missingness
    (:math:`\Pr(R=1 \mid A,W,S=s)`). The inverse probability weights are defined as

    .. math::

        \omega_i = \frac{1}{\Pr(A=a \mid W_i, S_i; \alpha)}
        \times \left[S_i + (1-S_i) \times \frac{\Pr(S=1 \mid W_i; \gamma)}{\Pr(S=0 \mid W_i; \gamma)} \right]
        \times \frac{1}{\Pr(R=1 \mid A_i, W_i, S_i; \rho)}

    These weights are then used to fit a weighted regression model for the outcome, :math:`E[Y \mid A,W,S,R=1]`. This
    model is then used to generate predictions under the different actions for the target population, :math:`S=1`. The
    AIPW estimator for :math:`A:=a` is defined as

    .. math::

        \hat{\mu}_{a,s} = \frac{\sum_{i=1}^{n} S_i \times  m_a(W, S=s; \hat{\beta})}{\sum_{i=1}^{n} S_i}

    where :math:`m_a(W,S=s; \beta)` is the predicted value from the weighted outcome model fit using :math:`S=1` given
    :math:`W` and :math:`A:=a`.

    The multi-span expression indicates that :math:`E[Y^1 | S=1]` can be separately estimated using the :math:`S=1`
    and :math:`S=0` data (i.e., :math:`\hat{\mu}_{1,1}` and :math:`\hat{\mu}_{1,0}`). Given the identification
    assumptions are met, a difference of zero between these estimators is expected . A non-zero difference is
    indicative of at least one assumption being violated. See the papers in the references for further details.

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
        # initialize the preceding class (this allows for more arguments in init than available in BaseMeanEstimator)
        super().__init__(data=data, outcome=outcome, action=action, sample=sample, alpha=alpha,
                         verbose=verbose, decimals=decimals)
        self.__estimator_label__ = "Augmented Inverse Probability Weighting"

    def estimate(self, init=None):
        """Estimate the parameters of interest using bridge augmented inverse probability weighting estimator.

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

    def diagnostics(self):
        """

        Returns
        -------

        """
        self.diagnostics_weights()
        print("")
        self.diagnostics_outcome()
        print("")
        self.diagnostic_shared()
