import warnings
import numpy as np
import pandas as pd
from delicatessen import MEstimator

from cantilever.estimators.time_to_event.basics import BridgeTimeEstimator
from cantilever.estimators.time_to_event.efuncs import (ef_risk_ipw, ef_diagnostic_ipw,
                                                        ef_single_span_ipw, ef_multi_span_ipw)
# TODO add diagnostic based on construct_weights, will require some efunc retooling


class BridgeIPW(BridgeTimeEstimator):
    r"""IPW bridge algorithm

    """
    def __init__(self, data, time, delta, action, sample, censor=None, alpha=0.05,
                 verbose=True, decimals=2):
        # initialize the preceding class (allows for more arguments in init than BaseEstimator)
        super().__init__(data=data, time=time, delta=delta, action=action, sample=sample, censor=censor,
                         alpha=alpha, verbose=verbose, decimals=decimals)

        # Updating specific parameters for g-computation
        self.__estimator_label__ = "Inverse Probability Weighting"

    def outcome_model(self, model):
        warnings.warn("`BridgeIPW` does not use the outcome model. No model was fit.",
                      UserWarning)

    def estimate_risks(self):
        if self._sample_nuisance_model_ is None:
            raise ValueError("The function sample_model() must be called prior to estimating the risks")
        if self._action_nuisance_model_ is None:
            raise ValueError("The function action_model() must be called prior to estimating the risks")

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
                return ef_risk_ipw(theta=theta, s=s, a=a, c=c, delta=delta,
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
                                   contribute_s=(s == samp),
                                   unique_censor_times=u_c_times,
                                   unique_event_times=u_times,
                                   product_limit=True)

            # Estimating weighted product limit
            init_risk = self._generate_inits_risk_(s=samp, a=act)
            nuisance_inits = self._sample_coefs_ + self._action_coefs_ + starting_censor
            starting_vals = init_risk + nuisance_inits
            estr = self._fit_mestimator_(psi, init=starting_vals, subset=list(range(n_unique_times)))
            est = estr.theta
            se = np.diag(estr.variance)**0.5
            ci = estr.confidence_intervals()

            # Storing results
            label = 'R-A' + str(act) + 'S' + str(samp)
            results[label] = est[: n_unique_times]
            results[label + '_SE'] = se[: n_unique_times]
            results[label + '_LCL'] = ci[: n_unique_times, 0]
            results[label + '_UCL'] = ci[: n_unique_times, 1]

        # Formatting results
        r0 = pd.DataFrame({"Time": [0, ],
                           "R-A2S1": [0., ], "R-A2S1_SE": [0., ], "R-A2S1_LCL": [0., ], "R-A2S1_UCL": [0., ],
                           "R-A1S1": [0., ], "R-A1S1_SE": [0., ], "R-A1S1_LCL": [0., ], "R-A1S1_UCL": [0., ],
                           "R-A1S0": [0., ], "R-A1S0_SE": [0., ], "R-A1S0_LCL": [0., ], "R-A1S0_UCL": [0., ],
                           "R-A0S0": [0., ], "R-A0S0_SE": [0., ], "R-A0S0_LCL": [0., ], "R-A0S0_UCL": [0., ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.risks = results.set_index("Time")

    def estimate_diagnostic(self):
        t, delta, a, s, c = self._get_variable_arrays_()
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
            return ef_diagnostic_ipw(theta=theta, s=s, a=a, c=c, delta=delta,
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
                                     unique_event_times=u_times,
                                     product_limit=True)

        if self.risks is not None:
            starting_vals = (list(self.risks['R-A1S1'] - self.risks['R-A1S0'])
                             + list(self.risks['R-A1S1'])[1:] + list(self.risks['R-A1S0'])[1:]
                             + self._sample_coefs_ + self._action_coefs_ + starting_censor)
            subset_solve = list(range(n_unique_times*3 + 1))
            # TODO can I modify this to reduce what we are searching over?
        else:
            starting_vals = ([0., ] + [0., ]*n_unique_times*3
                             + list(self._sample_coefs_) + list(self._action_coefs_) + starting_censor)
            subset_solve = list(range(n_unique_times*3 + 1))

        # subset_solve = list(range(n_unique_times*3 + 1))
        estr = self._fit_mestimator_(psi_diagnostic, init=starting_vals, subset=subset_solve)
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
        t, delta, a, s, c = self._get_variable_arrays_()
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

        def psi_singlespan(theta):
            return ef_single_span_ipw(theta=theta, s=s, a=a, c=c, delta=delta,
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
                                      unique_event_times=u_times,
                                      product_limit=True)

        if self.risks is None:
            starting_vals = ([0., ]*n_unique_times + [0.5, ]*n_unique_times*2
                             + self._sample_coefs_ + self._action_coefs_ + starting_censor)
        else:
            starting_vals = ([0., ]*n_unique_times + list(self.risks['R-A2S1'])[1:] + list(self.risks['R-A0S0'])[1:]
                             + self._sample_coefs_ + self._action_coefs_ + starting_censor)

        subset_solve = list(range(n_unique_times))
        estr = self._fit_mestimator_(psi_singlespan, init=starting_vals, subset=subset_solve)
        estr.estimate()
        est = estr.theta
        ci = estr.confidence_intervals()
        pval = estr.p_values()

        # Formatting results
        results = pd.DataFrame()
        results['Time'] = u_times
        results['RD-SS'] = est[:n_unique_times]
        results['RD-SS_LCL'] = ci[:n_unique_times, 0]
        results['RD-SS_UCL'] = ci[:n_unique_times, 1]
        results['RD-SS_P'] = pval[:n_unique_times]

        # Adding time zero to output
        r0 = pd.DataFrame({"Time": [0, ], "RD-SS": [0., ], "RD-SS_LCL": [0., ], "RD-SS_UCL": [0., ], 'RD-SS_P': [1, ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.single_span = results.set_index("Time")

    def estimate_multi_span(self):
        t, delta, a, s, c = self._get_variable_arrays_()
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

        def psi_multispan(theta):
            return ef_multi_span_ipw(theta=theta, s=s, a=a, c=c, delta=delta,
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
                                     unique_event_times=u_times,
                                     product_limit=True)

        if self.risks is None:
            starting_vals = ([0., ]*n_unique_times*5 + self._sample_coefs_ + self._action_coefs_ + starting_censor)
        else:
            starting_vals = ([0., ]*n_unique_times
                             + list(self.risks['R-A2S1'])[1:] + list(self.risks['R-A1S1'])[1:]
                             + list(self.risks['R-A1S0'])[1:] + list(self.risks['R-A0S0'])[1:]
                             + self._sample_coefs_ + self._action_coefs_ + starting_censor)

        subset_solve = list(range(n_unique_times * 5))
        estr = self._fit_mestimator_(psi_multispan, init=starting_vals, subset=subset_solve)
        est = estr.theta
        ci = estr.confidence_intervals()
        pval = estr.p_values()

        # Formatting results
        results = pd.DataFrame()
        results['Time'] = u_times
        results['RD-MS'] = est[0: n_unique_times]
        results['RD-MS_LCL'] = ci[0: n_unique_times, 0]
        results['RD-MS_UCL'] = ci[0: n_unique_times, 1]
        results['RD-MS_P'] = pval[0: n_unique_times]

        # Adding time zero to output
        r0 = pd.DataFrame({"Time": [0, ], "RD-MS": [0., ], "RD-MS_LCL": [0., ], "RD-MS_UCL": [0., ], 'RD-MS_P': [1, ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.multi_span = results.set_index("Time")
