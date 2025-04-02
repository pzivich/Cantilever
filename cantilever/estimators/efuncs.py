import numpy as np
from delicatessen.utilities import inverse_logit, identity

from cantilever.estimators.point.basics import psi_outcome, psi_action, psi_sample, psi_missing


def psi_bridge_point(theta, y, a, s, m, X, Xa2, Xa1, Xa0, out_model, Z, V, W, a_clip, s_clip, m_clip, include_missing,
                     aipw_implementation):
    """Stacked estimating functions for the bridged treatment comparison with a point action and outcome.

    Parameters
    ----------
    theta
    y
    a
    s
    m
    X
    Xa2
    Xa1
    Xa0
    out_model
    Z
    V
    W
    a_clip
    s_clip
    m_clip
    include_missing
    aipw_implementation

    Returns
    -------

    """
    # Breaking parameters into their models
    bridges = theta[0:3]     # Bridging parameters
    mu = theta[3:7]          # Causal means
    eta = theta[7:]          # Nuisance parameters

    # Dividing nuisance parameters based on what is input for psi
    if X is not None and Z is not None:
        n_out_params = X.shape[1]
        n_act_params = Z.shape[1] * 2
        n_smp_params = n_act_params + V.shape[1]
        beta = eta[:2*n_out_params]
        beta1 = beta[:n_out_params]
        beta0 = beta[n_out_params:]
        rho = eta[2*n_out_params:]
        alpha = rho[:n_act_params]
        gamma = rho[n_act_params:n_smp_params]
    elif X is not None:
        n_out_params = X.shape[1]
        beta = eta
        beta1 = beta[:n_out_params]
        beta0 = beta[n_out_params:]
    elif Z is not None:
        n_act_params = Z.shape[1] * 2
        n_smp_params = n_act_params + V.shape[1]
        rho = eta
        alpha = rho[:n_act_params]
        gamma = rho[n_act_params:n_smp_params]
    else:
        raise ValueError("Invalid model input. This error should never actually be reached due to prior errors that "
                         "should arise.")

    if Z is not None:
        # Action probability
        pi_a = (s * inverse_logit(np.dot(Z, alpha[:Z.shape[1]]))
                + (1-s) * inverse_logit(np.dot(Z, alpha[Z.shape[1]:])))
        pi_a = np.clip(pi_a, a_min=a_clip[0], a_max=a_clip[1])
        pi_a = ((a == 0) * (1 - pi_a) + (1 - s) * (a == 1) * pi_a + s * (a == 1) * (1 - pi_a) + (a == 2) * pi_a)

        # Sampling probability
        pi_s = inverse_logit(np.dot(V, gamma))
        pi_s = np.clip(pi_s, a_min=s_clip[0], a_max=s_clip[1])
        pi_s = s * 1 + (1 - s) * (1 - pi_s) / pi_s

        # Missing probability
        if include_missing:
            delta = rho[n_smp_params:]
            pi_m = inverse_logit(np.dot(W, delta))
            pi_m = np.clip(pi_m, a_min=m_clip[0], a_max=m_clip[1])
            pi = pi_a * pi_s * pi_m
        else:
            pi = pi_a * pi_s
        ipw = 1 / pi

    # Storage for nuisance model estimating equations output
    ee_nuisance = []

    # Outcome nuisance models
    if X is not None:
        if out_model == 'linear':
            transform = identity
        elif out_model == 'logistic':
            transform = inverse_logit
        elif out_model == 'poisson':
            transform = np.exp
        else:
            raise ValueError("The outcome model specification " + str(out_model) + " is not supported.")
        if Z is not None and aipw_implementation.lower() == "weighted-regression":
            ee_out = psi_outcome(theta=beta, X=X, y=y, s=s, m=m, model=out_model, weights=ipw)
        else:
            ee_out = psi_outcome(theta=beta, X=X, y=y, s=s, m=m, model=out_model, weights=None)
        y2hat = transform(np.dot(Xa2, beta1))
        y1ahat = transform(np.dot(Xa1, beta1))
        y1bhat = transform(np.dot(Xa1, beta0))
        y0hat = transform(np.dot(Xa0, beta0))
        ee_nuisance.append(ee_out)

    # Inverse Probability Weight Calculations
    if Z is not None:
        pass
        # Action score nuisance model
        ee_act = psi_action(theta=alpha, Z=Z, a=a, s=s)
        ee_nuisance.append(ee_act)

        # Sampling score nuisance model
        ee_smp = psi_sample(theta=gamma, V=V, s=s)
        ee_nuisance.append(ee_smp)

        # Missingness score nuisance model
        if include_missing:
            ee_mis = psi_missing(theta=delta, W=W, m=m)
            ee_nuisance.append(ee_mis)
        else:
            pass

    # Bridge estimating functions
    ee_ss = np.ones(y.shape[0]) * (mu[0] - mu[3]) - bridges[0]
    ee_ms = np.ones(y.shape[0]) * (mu[0] - mu[1] + mu[2] - mu[3]) - bridges[1]
    ee_diag = np.ones(y.shape[0]) * (mu[1] - mu[2]) - bridges[2]

    # Causal mean estimating functions
    if X is None:
        ee_mean12 = s * m * (a == 2) * ipw * (y - mu[0])
        ee_mean11 = s * m * (a == 1) * ipw * (y - mu[1])
        ee_mean01 = (1 - s) * m * (a == 1) * ipw * (y - mu[2])
        ee_mean00 = (1 - s) * m * (a == 0) * ipw * (y - mu[3])
    elif Z is None or aipw_implementation.lower() == "weighted-regression":
        ee_mean12 = s * (y2hat - mu[0])
        ee_mean11 = s * (y1ahat - mu[1])
        ee_mean01 = s * (y1bhat - mu[2])
        ee_mean00 = s * (y0hat - mu[3])
    elif aipw_implementation.lower() == "classic":
        ee_mean12 = s * ((y*m*(a == 2)*ipw - y2hat*((a == 2)-pi)/pi) - mu[0])
        ee_mean11 = s * ((y*m*(a == 1)*ipw + y1ahat*((a == 1)-pi)/(1-pi)) - mu[1])
        ee_mean01 = (1-s) * m * (a == 1) * ipw * (y - y1bhat) + s*(y1bhat - mu[2])
        ee_mean00 = (1-s) * m * (a == 0) * ipw * (y - y0hat) + s*(y0hat - mu[3])
        pass
    else:
        raise ValueError("Invalid AIPW specification. This error should never actually be reached due to prior errors "
                         "that should arise.")

    # Returning the estimating functions
    return np.vstack([ee_ss, ee_ms, ee_diag,
                      ee_mean12, ee_mean11, ee_mean01, ee_mean00, ] + ee_nuisance)
