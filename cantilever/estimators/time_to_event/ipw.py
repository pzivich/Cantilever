import warnings
import numpy as np
import pandas as pd
from delicatessen import MEstimator

from cantilever.estimators.time_to_event.basics import BridgeTimeEstimator
from cantilever.estimators.time_to_event.efuncs import (ef_risk_product_limit,
                                                        ef_diagnostic_product_limit,
                                                        ef_diagnostic_product_limit_show, product_limit_predict
                                                        )


class BridgeIPW(BridgeTimeEstimator):
    r"""IPW bridge algorithm

    """
    def __init__(self, data, time, delta, action, sample, alpha=0.05, verbose=True, decimals=2):
        # initialize the preceding class (allows for more arguments in init than BaseEstimator)
        super().__init__(data=data, time=time, delta=delta, action=action, sample=sample,
                         alpha=alpha, verbose=verbose, decimals=decimals)

        # Updating specific parameters for g-computation
        self.__estimator_label__ = "Inverse Probability Weighting"

    def outcome_model(self, model):
        warnings.warn("`BridgeIPW` does not use the outcome model. No model was fit.",
                      UserWarning)

    def estimate_risks(self):
        t, delta, a, s = self._get_variable_arrays_()
        t_matrix, f_matrix, r_matrix, u_times = self._get_time_matrices_(t=t, action=None, sample=None)
        n_unique_times = len(self.all_unique_times)

        # Fitting a series of weighted product limits
        results = pd.DataFrame()
        results['Time'] = self.all_unique_times
        for act_samp_combo in self._combinations_:
            act = act_samp_combo[0]
            samp = act_samp_combo[1]

            def psi(theta):
                return ef_risk_product_limit(theta=theta,
                                             s=s,
                                             a=a,
                                             delta=delta,
                                             sample_matrix=self._sample_design_matrix_,
                                             action_matrix=self._action_design_matrix_,
                                             final_time_matrix=f_matrix,
                                             risk_set_matrix=r_matrix,
                                             contribute=(s == samp) & (a == act))

            # Estimating weighted product limit
            starting_vals = [0.01, ]*n_unique_times + self._sample_coefs_ + self._action_coefs_
            estr = MEstimator(psi, init=starting_vals)
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
        t, delta, a, s = self._get_variable_arrays_()
        # TODO need to add 0, max_t to _get_time_matrices_ for the diagnostic, so IRD is correct
        t_matrix, f_matrix, r_matrix, u_times = self._get_time_matrices_(t=t, action=None, sample=None)
        n_unique_times = len(u_times)

        def psi_diagnostic(theta):
            return ef_diagnostic_product_limit(theta=theta,
                                               s=s,
                                               a=a,
                                               delta=delta,
                                               sample_matrix=self._sample_design_matrix_,
                                               action_matrix=self._action_design_matrix_,
                                               final_time_matrix=f_matrix,
                                               risk_set_matrix=r_matrix,
                                               contribute_s1=(a == 1) & (s == 1),
                                               contribute_s0=(a == 1) & (s == 0))

        if self.risks is None:
            starting_vals = [0., ] + [0., ]*n_unique_times*3 + self._sample_coefs_ + self._action_coefs_
        else:
            starting_vals = ([0., ] + [0., ]*n_unique_times
                             + list(self.risks['R-A1S1'])[1:] + list(self.risks['R-A1S0'])[1:]
                             + self._sample_coefs_ + self._action_coefs_)

        estr = MEstimator(psi_diagnostic, init=starting_vals)
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
