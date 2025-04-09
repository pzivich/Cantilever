import warnings
import numpy as np
import pandas as pd
from delicatessen import MEstimator

from cantilever.formulas import get_design_matrix
from cantilever.estimators.time_to_event.basics import BridgeTimeEstimator
from cantilever.estimators.time_to_event.efuncs import (ef_risk_gcomp, ef_diagnostic_gcomp,
                                                        ef_single_span_gcomp, ef_multi_span_gcomp)


class BridgeGComputation(BridgeTimeEstimator):
    r"""G-computation bridge algorithm

    """
    def __init__(self, data, time, delta, action, sample, censor=None, alpha=0.05, verbose=True, decimals=2):
        # initialize the preceding class (allows for more arguments in init than BaseEstimator)
        super().__init__(data=data, time=time, delta=delta, action=action, sample=sample, censor=censor,
                         alpha=alpha, verbose=verbose, decimals=decimals)

        # Updating specific parameters for g-computation
        self.__estimator_label__ = "Parametric G-computation"

    def action_model(self, model, init=None, bounds=(0, 1)):
        warnings.warn("`BridgeGComputation` does not use the action model. No model was fit.",
                      UserWarning)

    def sample_model(self, model, init=None, bounds=(0, 1)):
        warnings.warn("`BridgeGComputation` does not use the sample model. No model was fit.",
                      UserWarning)

    def censor_model(self, model, init=None, bounds=(0, 1)):
        warnings.warn("`BridgeGComputation` does not use the censoring model. No model was fit.",
                      UserWarning)

    def estimate_risks(self):
        if self._outcome_nuisance_model_ is None:
            raise ValueError("The function outcome_model() must be called prior to estimating the risks")

        n_unique_times = len(self.all_unique_times)
        t, delta, a, s, c = self._get_variable_arrays_()
        matrix, col_names = get_design_matrix(self._outcome_nuisance_model_, self.data)
        baseline_matrix = np.asarray(matrix)

        # Fitting a series of pooled logistic models
        results = pd.DataFrame()
        results['Time'] = self.all_unique_times
        for i in range(len(self._combinations_)):
            act_samp_combo = self._combinations_[i]
            act = act_samp_combo[0]
            samp = act_samp_combo[1]
            t_matrix, f_matrix, r_matrix, u_times = self._get_time_matrices_(t=t, action=act, sample=samp)
            # TODO add weights

            def psi(theta):
                return ef_risk_gcomp(theta=theta,
                                     delta=delta,
                                     baseline_matrix=baseline_matrix,
                                     time_matrix=t_matrix,
                                     final_time_matrix=f_matrix,
                                     risk_set_matrix=r_matrix,
                                     contribute=(a == act) & (s == samp),
                                     strata_unique_times=u_times,
                                     all_unique_times=self.all_unique_times)

            # Estimating pooled logistic model
            starting_vals = [0.01, ]*n_unique_times + self._outcome_coefs_[i]
            estr = MEstimator(psi, init=starting_vals, subset=list(range(n_unique_times)))
            estr.estimate()
            est = estr.theta
            ci = estr.confidence_intervals()

            # Storing results
            label = 'R-A' + str(act) + 'S' + str(samp)
            results[label] = est[: n_unique_times]
            results[label+'_LCL'] = ci[: n_unique_times, 0]
            results[label+'_UCL'] = ci[: n_unique_times, 1]

        # Formatting results
        r0 = pd.DataFrame({"Time": [0, ],
                           "R-A2S1": [0., ], "R-A2S1_LCL": [0., ], "R-A2S1_UCL": [0., ],
                           "R-A1S1": [0., ], "R-A1S1_LCL": [0., ], "R-A1S1_UCL": [0., ],
                           "R-A1S0": [0., ], "R-A1S0_LCL": [0., ], "R-A1S0_UCL": [0., ],
                           "R-A0S0": [0., ], "R-A0S0_LCL": [0., ], "R-A0S0_UCL": [0., ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.risks = results.set_index("Time")

    def estimate_diagnostic(self):
        if self._outcome_nuisance_model_ is None:
            raise ValueError("The function outcome_model() must be called prior to estimating the diagnostic")

        n_unique_times = len(self.all_unique_times)
        t, delta, a, s, c = self._get_variable_arrays_()
        matrix, col_names = get_design_matrix(self._outcome_nuisance_model_, self.data)
        baseline_matrix = np.asarray(matrix)
        t_matrix1, f_matrix1, r_matrix1, u_times1 = self._get_time_matrices_(t=t, action=1, sample=1)
        t_matrix0, f_matrix0, r_matrix0, u_times0 = self._get_time_matrices_(t=t, action=1, sample=0)

        def psi_diagnostic(theta):
            return ef_diagnostic_gcomp(theta=theta, delta=delta, sample=s,
                                       baseline_matrix=baseline_matrix,
                                       time_matrix_s1=t_matrix1,
                                       final_time_matrix_s1=f_matrix1,
                                       risk_set_matrix_s1=r_matrix1,
                                       contribute_s1=(a == 1) * (s == 1),
                                       strata_s1_times=u_times1,
                                       time_matrix_s0=t_matrix0,
                                       final_time_matrix_s0=f_matrix0,
                                       risk_set_matrix_s0=r_matrix0,
                                       contribute_s0=(a == 1) * (s == 0),
                                       strata_s0_times=u_times0,
                                       all_unique_times=self.all_unique_times)

        starting_vals = [0., ] + [0., ]*n_unique_times + self._outcome_coefs_[1] + self._outcome_coefs_[2]
        estr = MEstimator(psi_diagnostic, init=starting_vals, subset=list(range(n_unique_times+1)))
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
        results['Time'] = self.all_unique_times
        results['RD-D'] = est[1: n_unique_times + 1]
        results['RD-D_LCL'] = ci[1: n_unique_times + 1, 0]
        results['RD-D_UCL'] = ci[1: n_unique_times + 1, 1]
        results['RD-D_P'] = pval[1: n_unique_times + 1]

        # Adding time zero to output
        r0 = pd.DataFrame({"Time": [0, ], "RD-D": [0., ], "RD-D_LCL": [0., ], "RD-D_UCL": [0., ], 'RD-D_P': [1, ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.diagnostic = results.set_index("Time")

    def estimate_single_span(self):
        if self._outcome_nuisance_model_ is None:
            raise ValueError("The function outcome_model() must be called prior to estimating the risk difference")

        n_unique_times = len(self.all_unique_times)
        t, delta, a, s, c = self._get_variable_arrays_()
        matrix, col_names = get_design_matrix(self._outcome_nuisance_model_, self.data)
        baseline_matrix = np.asarray(matrix)
        t_matrix1, f_matrix1, r_matrix1, u_times1 = self._get_time_matrices_(t=t, action=2, sample=1)
        t_matrix0, f_matrix0, r_matrix0, u_times0 = self._get_time_matrices_(t=t, action=0, sample=0)
        # TODO add weights

        def psi_single_span(theta):
            return ef_single_span_gcomp(theta=theta, delta=delta, sample=s,
                                        baseline_matrix=baseline_matrix,
                                        time_matrix_s1=t_matrix1,
                                        final_time_matrix_s1=f_matrix1,
                                        risk_set_matrix_s1=r_matrix1,
                                        contribute_s1=(a == 2) * (s == 1),
                                        strata_s1_times=u_times1,
                                        time_matrix_s0=t_matrix0,
                                        final_time_matrix_s0=f_matrix0,
                                        risk_set_matrix_s0=r_matrix0,
                                        contribute_s0=(a == 0) * (s == 0),
                                        strata_s0_times=u_times0,
                                        all_unique_times=self.all_unique_times)

        start_vals = [0., ] * n_unique_times + self._outcome_coefs_[0] + self._outcome_coefs_[3]
        estr = MEstimator(psi_single_span, init=start_vals, subset=list(range(n_unique_times)))
        estr.estimate()
        est = estr.theta
        ci = estr.confidence_intervals()
        pval = estr.p_values()

        # Formatting results
        results = pd.DataFrame()
        results['Time'] = self.all_unique_times
        results['RD-SS'] = est[:n_unique_times]
        results['RD-SS_LCL'] = ci[:n_unique_times, 0]
        results['RD-SS_UCL'] = ci[:n_unique_times, 1]
        results['RD-SS_P'] = pval[:n_unique_times]

        # Adding time zero to output
        r0 = pd.DataFrame({"Time": [0, ], "RD-SS": [0., ], "RD-SS_LCL": [0., ], "RD-SS_UCL": [0., ], 'RD-SS_P': [1, ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.single_span = results.set_index("Time")

    def estimate_multi_span(self):
        if self._outcome_nuisance_model_ is None:
            raise ValueError("The function outcome_model() must be called prior to estimating the risk difference")

        n_unique_times = len(self.all_unique_times)
        t, delta, a, s, c = self._get_variable_arrays_()
        matrix, col_names = get_design_matrix(self._outcome_nuisance_model_, self.data)
        baseline_matrix = np.asarray(matrix)
        t_matrix3, f_matrix3, r_matrix3, u_times3 = self._get_time_matrices_(t=t, action=2, sample=1)
        t_matrix2, f_matrix2, r_matrix2, u_times2 = self._get_time_matrices_(t=t, action=1, sample=1)
        t_matrix1, f_matrix1, r_matrix1, u_times1 = self._get_time_matrices_(t=t, action=1, sample=0)
        t_matrix0, f_matrix0, r_matrix0, u_times0 = self._get_time_matrices_(t=t, action=0, sample=0)
        # TODO add weights

        def psi_multi_span(theta):
            return ef_multi_span_gcomp(theta=theta, delta=delta, sample=s,
                                       baseline_matrix=baseline_matrix,
                                       time_matrix_s3=t_matrix3,
                                       final_time_matrix_s3=f_matrix3,
                                       risk_set_matrix_s3=r_matrix3,
                                       contribute_s3=(a == 2) * (s == 1),
                                       strata_s3_times=u_times3,
                                       time_matrix_s2=t_matrix2,
                                       final_time_matrix_s2=f_matrix2,
                                       risk_set_matrix_s2=r_matrix2,
                                       contribute_s2=(a == 1) * (s == 1),
                                       strata_s2_times=u_times2,
                                       time_matrix_s1=t_matrix1,
                                       final_time_matrix_s1=f_matrix1,
                                       risk_set_matrix_s1=r_matrix1,
                                       contribute_s1=(a == 1) * (s == 0),
                                       strata_s1_times=u_times1,
                                       time_matrix_s0=t_matrix0,
                                       final_time_matrix_s0=f_matrix0,
                                       risk_set_matrix_s0=r_matrix0,
                                       contribute_s0=(a == 0) * (s == 0),
                                       strata_s0_times=u_times0,
                                       all_unique_times=self.all_unique_times)

        start_vals = ([0., ] * n_unique_times
                      + self._outcome_coefs_[0] + self._outcome_coefs_[1]
                      + self._outcome_coefs_[2] + self._outcome_coefs_[3])
        estr = MEstimator(psi_multi_span, init=start_vals, subset=list(range(n_unique_times)))
        estr.estimate()
        est = estr.theta
        ci = estr.confidence_intervals()
        pval = estr.p_values()

        # Formatting results
        results = pd.DataFrame()
        results['Time'] = self.all_unique_times
        results['RD-MS'] = est[:n_unique_times]
        results['RD-MS_LCL'] = ci[:n_unique_times, 0]
        results['RD-MS_UCL'] = ci[:n_unique_times, 1]
        results['RD-MS_P'] = pval[:n_unique_times]

        # Adding time zero to output
        r0 = pd.DataFrame({"Time": [0, ], "RD-MS": [0., ], "RD-MS_LCL": [0., ], "RD-MS_UCL": [0., ], 'RD-MS_P': [1, ]})
        results = pd.concat([r0, results], ignore_index=True)
        self.multi_span = results.set_index("Time")
