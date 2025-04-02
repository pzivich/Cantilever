import numpy as np
import pandas as pd
from delicatessen import MEstimator


def fit_mestimator(estimating_functions, init, solver, maxiter, tolerance, deriv_method, dx, subset=None):
    estr = MEstimator(estimating_functions, init=init, subset=subset)
    estr.estimate(solver=solver, maxiter=maxiter, tolerance=tolerance,
                  deriv_method=deriv_method, dx=dx, allow_pinv=True)
    return estr


def print_nuisance_model_results(labels, m_estimator, decimals=3):
    r = pd.DataFrame()
    r['_'] = labels
    r['_'] = r['_'].str.slice(0, 25)
    r['coef'] = m_estimator.theta
    r['stderr'] = np.diag(m_estimator.variance)**0.5
    r['Z-value'] = m_estimator.z_scores()
    conf_int = m_estimator.confidence_intervals()
    r['LCL'] = conf_int[:, 0]
    r['UCL'] = conf_int[:, 1]
    r = r.set_index("_")
    print(r.round(decimals))

