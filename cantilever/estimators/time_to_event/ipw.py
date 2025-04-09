import warnings
from time import time
import numpy as np
import pandas as pd
from delicatessen import MEstimator

from cantilever.estimators.time_to_event.basics import BridgeTimeEstimator
from cantilever.estimators.time_to_event.efuncs import (ef_risk_product_limit,
                                                        ef_diagnostic_product_limit,
                                                        product_limit_predict, construct_weights
                                                        )


class BridgeIPW(BridgeTimeEstimator):
    r"""IPW bridge algorithm

    """
    def __init__(self, data, time, delta, action, sample, censor=None, alpha=0.05, verbose=True, decimals=2):
        # initialize the preceding class (allows for more arguments in init than BaseEstimator)
        super().__init__(data=data, time=time, delta=delta, action=action, sample=sample, censor=censor,
                         alpha=alpha, verbose=verbose, decimals=decimals)

        # Updating specific parameters for g-computation
        self.__estimator_label__ = "Inverse Probability Weighting"

    def outcome_model(self, model):
        warnings.warn("`BridgeIPW` does not use the outcome model. No model was fit.",
                      UserWarning)

    def estimate_risks(self):
        t, delta, a, s, c = self._get_variable_arrays_()
        t_matrix, f_matrix, r_matrix, u_times = self._get_time_matrices_(t=t, action=None, sample=None)
        n_unique_times = len(self.all_unique_times)

        # Fitting a series of weighted product limits
        results = pd.DataFrame()
        results['Time'] = self.all_unique_times
        for act_samp_combo in self._combinations_:
            act = act_samp_combo[0]
            samp = act_samp_combo[1]
            if self._censor_design_matrix_ is None:
                t1_matrix, f1_matrix, r1_matrix, u1_times = None, None, None, None
                t0_matrix, f0_matrix, r0_matrix, u0_times = None, None, None, None
                u_c_times, u_d_times = None, None
                starting_censor = []
            else:
                t1_matrix, f1_matrix, r1_matrix, u1_times = self._get_censor_matrices_(t=t, c=c, sample=1)
                t0_matrix, f0_matrix, r0_matrix, u0_times = self._get_censor_matrices_(t=t, c=c, sample=0)
                u_c_times = list(np.unique(self.data.loc[c == 1, self.time]))
                starting_censor = self._censor_coefs_[0] + self._censor_coefs_[1]

            def psi(theta):
                return ef_risk_product_limit(theta=theta,
                                             s=s,
                                             a=a,
                                             c=c,
                                             delta=delta,
                                             sample_matrix=self._sample_design_matrix_,
                                             action_matrix=self._action_design_matrix_,
                                             censor_matrix=self._censor_design_matrix_,
                                             time_matrix_s1=t1_matrix,
                                             final_time_matrix_s1=f1_matrix,
                                             risk_set_matrix_s1=r1_matrix,
                                             strata_s1_times=u1_times,
                                             time_matrix_s0=t0_matrix,
                                             final_time_matrix_s0=f0_matrix,
                                             risk_set_matrix_s0=r0_matrix,
                                             strata_s0_times=u0_times,
                                             final_time_matrix=f_matrix,
                                             risk_set_matrix=r_matrix,
                                             contribute=(s == samp) & (a == act),
                                             unique_censor_times=u_c_times,
                                             unique_event_times=u_times)

            # Estimating weighted product limit
            starting_vals = [0.01, ]*n_unique_times + self._sample_coefs_ + self._action_coefs_ + starting_censor
            estr = MEstimator(psi, init=starting_vals, subset=list(range(n_unique_times+1)))
            estr.estimate()
            est = estr.theta
            ci = estr.confidence_intervals()

            # Storing results
            label = 'R-A' + str(act) + 'S' + str(samp)
            results[label] = est[: n_unique_times]
            results[label + '_LCL'] = ci[: n_unique_times, 0]
            results[label + '_UCL'] = ci[: n_unique_times, 1]

        # Formatting results
        r0 = pd.DataFrame({"Time": [0, ],
                           "R-A2S1": [0., ], "R-A2S1_LCL": [0., ], "R-A2S1_UCL": [0., ],
                           "R-A1S1": [0., ], "R-A1S1_LCL": [0., ], "R-A1S1_UCL": [0., ],
                           "R-A1S0": [0., ], "R-A1S0_LCL": [0., ], "R-A1S0_UCL": [0., ],
                           "R-A0S0": [0., ], "R-A0S0_LCL": [0., ], "R-A0S0_UCL": [0., ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.risks = results.set_index("Time")

    def estimate_diagnostic(self):
        t, delta, a, s, c = self._get_variable_arrays_()
        # TODO need to add 0, max_t to _get_time_matrices_ for the diagnostic, so IRD is correct
        t_matrix, f_matrix, r_matrix, u_times = self._get_time_matrices_(t=t, action=None, sample=None)
        n_unique_times = len(u_times)
        if self._censor_design_matrix_ is None:
            t1_matrix, f1_matrix, r1_matrix, u1_times = None, None, None, None
            t0_matrix, f0_matrix, r0_matrix, u0_times = None, None, None, None
            u_c_times, u_d_times = None, None
            starting_censor = []
        else:
            t1_matrix, f1_matrix, r1_matrix, u1_times = self._get_censor_matrices_(t=t, c=c, sample=1)
            t0_matrix, f0_matrix, r0_matrix, u0_times = self._get_censor_matrices_(t=t, c=c, sample=0)
            u_c_times = list(np.unique(self.data.loc[c == 1, self.time]))
            starting_censor = self._censor_coefs_[0] + self._censor_coefs_[1]

        def psi_diagnostic(theta):
            return ef_diagnostic_product_limit(theta=theta,
                                               s=s,
                                               a=a,
                                               c=c,
                                               delta=delta,
                                               sample_matrix=self._sample_design_matrix_,
                                               action_matrix=self._action_design_matrix_,
                                               censor_matrix=self._censor_design_matrix_,
                                               time_matrix_s1=t1_matrix,
                                               final_time_matrix_s1=f1_matrix,
                                               risk_set_matrix_s1=r1_matrix,
                                               strata_s1_times=u1_times,
                                               time_matrix_s0=t0_matrix,
                                               final_time_matrix_s0=f0_matrix,
                                               risk_set_matrix_s0=r0_matrix,
                                               strata_s0_times=u0_times,
                                               final_time_matrix=f_matrix,
                                               risk_set_matrix=r_matrix,
                                               unique_censor_times=u_c_times,
                                               unique_event_times=u_times
                                               )

        if self.risks is None:
            starting_vals = ([0., ] + [0., ]*n_unique_times*3
                             + self._sample_coefs_ + self._action_coefs_ + starting_censor)
            subset_solve = list(range(0, n_unique_times*3 + 2))
        else:
            starting_vals = ([0., ] + [0., ]*n_unique_times
                             + list(self.risks['R-A1S1'])[1:] + list(self.risks['R-A1S0'])[1:]
                             + self._sample_coefs_ + self._action_coefs_ + starting_censor)
            subset_solve = list(range(0, n_unique_times + 2))

        estr = MEstimator(psi_diagnostic, init=starting_vals, subset=subset_solve)
        estr.estimate()
        est = estr.theta
        ci = estr.confidence_intervals()
        pval = estr.p_values()

        # Integrated risk difference results
        self.diagnostic_test = pd.DataFrame({"IRD": [estr.theta[0], ],
                                             "IRD_LCL": [ci[0, 0], ], "IRD_UCL": [ci[0, 1], ],
                                             "P-value": pval[0]})

        # Formatting results
        results = pd.DataFrame()
        results['Time'] = u_times
        results['RD-D'] = est[1: n_unique_times + 1]
        results['RD-D_LCL'] = ci[1: n_unique_times + 1, 0]
        results['RD-D_UCL'] = ci[1: n_unique_times + 1, 1]
        results['RD-D_P'] = pval[1: n_unique_times + 1]

        # Adding time zero to output
        r0 = pd.DataFrame({"Time": [0, ], "RD-D": [0., ], "RD-D_LCL": [0., ], "RD-D_UCL": [0., ], 'RD-D_P': [1, ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.diagnostic = results.set_index("Time")

    def estimate_single_span(self):
        pass

    def estimate_multi_span(self):
        pass
