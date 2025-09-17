import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.utilities import inverse_logit


def fit_mestimator(estimating_functions, init, solver, maxiter, tolerance, deriv_method, dx, subset=None):
    estr = MEstimator(estimating_functions, init=init, subset=subset)
    estr.estimate(solver=solver, maxiter=maxiter, tolerance=tolerance,
                  deriv_method=deriv_method, dx=dx, allow_pinv=True)
    return estr


def compute_action_score(a, s, param, design_Z, clip):
    id_as = design_Z.shape[1]
    pi_a = ((1 - s) * inverse_logit(np.dot(design_Z, param[:id_as]))
            + s * inverse_logit(np.dot(design_Z, param[id_as:])))
    pi_a = np.clip(pi_a, a_min=clip[0], a_max=clip[1])
    pi_a = ((a == 0) * (1 - pi_a) + (1 - s) * (a == 1) * pi_a + s * (a == 1) * (1 - pi_a) + (a == 2) * pi_a)
    return pi_a


def compute_sample_score(s, param, design_V, clip):
    pi_s = inverse_logit(np.dot(design_V, param))
    pi_s = np.clip(pi_s, a_min=clip[0], a_max=clip[1])
    pi_s = s*1 + (1 - s)*(1 - pi_s)/pi_s
    return pi_s


def compute_missing_score(param, design_W, clip):
    pi_m = inverse_logit(np.dot(design_W, param))
    pi_m = np.clip(pi_m, a_min=clip[0], a_max=clip[1])
    return pi_m


def descriptive_stats_w(var, values, w, label, include_overall=False, decimals=3):
    r = pd.DataFrame()
    r['_'] = ["Mean", "SD", "Min", "P25", "P50", "P75", "Max", "Sum"]
    if include_overall:
        r[label] = [np.mean(w), np.std(w, ddof=1),
                    np.min(w), np.percentile(w, q=25), np.percentile(w, q=50),
                    np.percentile(w, q=75), np.max(w), np.sum(w)]
    for v in values:
        wv = w[var == v]
        r[label+'='+str(v)] = [np.mean(wv), np.std(wv, ddof=1),
                               np.min(wv), np.percentile(wv, q=25), np.percentile(wv, q=50),
                               np.percentile(wv, q=75), np.max(wv), np.sum(wv)]

    r = r.set_index("_")
    print(r.round(decimals))


def descriptive_stats_r(study, residual, label, decimals=3):
    r = pd.DataFrame()
    r['_'] = ["Mean", "SD", "Min", "P25", "P50", "P75", "Max"]
    for v in [0, 1]:
        wv = residual[study == v]
        r[label+'='+str(v)] = [np.mean(wv), np.std(wv, ddof=1),
                               np.min(wv), np.percentile(wv, q=25), np.percentile(wv, q=50),
                               np.percentile(wv, q=75), np.max(wv)]

    r = r.set_index("_")
    print(r.round(decimals))


def print_nuisance_model_results(labels, m_estimator, decimals=3, subset=None):
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
    if subset is not None:
        r = r.iloc[:subset].copy()
    print(r.round(decimals))

