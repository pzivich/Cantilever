import numpy as np
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit

from cantilever.estimators.time_to_event.utils import align_ipcw_with_event_times


def ef_sample_logit(theta, s, sample_matrix):
    return ee_regression(theta, X=sample_matrix, y=s, model='logistic')


def ef_action_logit(theta, s, a, action_matrix):
    n_params = action_matrix.shape[1]
    beta1 = theta[:n_params]
    beta0 = theta[n_params:]
    ee_act1 = ee_regression(theta=beta1, X=action_matrix, y=a-1, model='logistic') * s
    ee_act0 = ee_regression(theta=beta0, X=action_matrix, y=a, model='logistic') * (1-s)
    return np.vstack([ee_act1, ee_act0])


def ef_weight_models(theta, s, a, c, sample_matrix, action_matrix, censor_matrix=None,
                     time_matrix_s1=None, final_time_matrix_s1=None, risk_set_matrix_s1=None,
                     time_matrix_s0=None, final_time_matrix_s0=None, risk_set_matrix_s0=None):
    if censor_matrix is None:
        m_index1 = sample_matrix.shape[1]
        beta = theta[:m_index1]
        gamma = theta[m_index1:]

        # Nuisance models
        ee_sample = ef_sample_logit(beta, s, sample_matrix)
        ee_action = ef_action_logit(gamma, s, a, action_matrix)
        return np.vstack([ee_sample, ee_action])
    else:
        m_index1 = sample_matrix.shape[1]
        m_index2 = m_index1 + action_matrix.shape[1]*2
        m_index3 = censor_matrix.shape[1] + time_matrix_s1.shape[0]
        beta = theta[:m_index1]
        gamma = theta[m_index1:m_index2]
        phi = theta[m_index2:]
        phi1 = phi[:m_index3]
        phi0 = phi[m_index3:]

        # Nuisance models
        ee_sample = ef_sample_logit(beta, s, sample_matrix)
        ee_action = ef_action_logit(gamma, s, a, action_matrix)
        ee_censor1 = ef_pooled_logit(theta=phi1, delta=c, baseline_matrix=censor_matrix,
                                     time_matrix=time_matrix_s1, final_time_matrix=final_time_matrix_s1,
                                     risk_set_matrix=risk_set_matrix_s1, contribute=(s == 1))
        ee_censor0 = ef_pooled_logit(theta=phi0, delta=c, baseline_matrix=censor_matrix,
                                     time_matrix=time_matrix_s0, final_time_matrix=final_time_matrix_s0,
                                     risk_set_matrix=risk_set_matrix_s0, contribute=(s == 0))
        return np.vstack([ee_sample, ee_action, ee_censor1, ee_censor0])


def construct_weights(theta, s, a, sample_matrix, action_matrix, censor_matrix=None,
                      time_matrix_s1=None, strata_s1_times=None, time_matrix_s0=None, strata_s0_times=None,
                      unique_censor_times=None, unique_event_times=None):
    if censor_matrix is None:
        m_index1 = sample_matrix.shape[1]
        beta = theta[:m_index1]
        gamma = theta[m_index1:]

        # Constructing the inverse odds of sampling weights
        pr_s = inverse_logit(np.dot(sample_matrix, beta))
        iosw = s + ((1-s)*(1-pr_s)/pr_s)

        # Constructing the inverse probability of treatment weights
        gamma1 = gamma[:action_matrix.shape[1]]
        gamma0 = gamma[action_matrix.shape[1]:]
        pr_a2 = inverse_logit(np.dot(action_matrix, gamma1))
        pr_a1 = inverse_logit(np.dot(action_matrix, gamma0))
        iptw = ((a == 2) / pr_a2
                + (a == 1)*(s == 1) / (1-pr_a2)
                + (a == 1)*(s == 0) / pr_a1
                + (a == 0) / (1-pr_a1))
        ipw = iosw * iptw
        return ipw
    else:
        m_index1 = sample_matrix.shape[1]
        m_index2 = m_index1 + action_matrix.shape[1]*2
        m_index3 = censor_matrix.shape[1] + time_matrix_s1.shape[0]
        beta = theta[:m_index1]
        gamma = theta[m_index1:m_index2]
        phi = theta[m_index2:]
        phi1 = phi[:m_index3]
        phi0 = phi[m_index3:]

        # Constructing the inverse odds of sampling weights
        pr_s = inverse_logit(np.dot(sample_matrix, beta))
        iosw = s + ((1 - s) * (1 - pr_s) / pr_s)

        # Constructing the inverse probability of treatment weights
        gamma1 = gamma[:action_matrix.shape[1]]
        gamma0 = gamma[action_matrix.shape[1]:]
        pr_a2 = inverse_logit(np.dot(action_matrix, gamma1))
        pr_a1 = inverse_logit(np.dot(action_matrix, gamma0))
        iptw = ((a == 2) / pr_a2
                + (a == 1) * (s == 1) / (1 - pr_a2)
                + (a == 1) * (s == 0) / pr_a1
                + (a == 0) / (1 - pr_a1))

        # Constructing the inverse probability of censoring weights
        pr_c1 = 1 - plogit_predictions(theta=phi1, baseline_matrix=censor_matrix, time_matrix=time_matrix_s1,
                                       strata_unique_times=strata_s1_times, all_unique_times=unique_censor_times)
        pr_c0 = 1 - plogit_predictions(theta=phi0, baseline_matrix=censor_matrix, time_matrix=time_matrix_s0,
                                       strata_unique_times=strata_s0_times, all_unique_times=unique_censor_times)
        ipcw = 1 / (pr_c1 * s + pr_c0 * (1 - s))

        # Aligning censor weights
        ipcw = align_ipcw_with_event_times(event_time_vector=unique_event_times, censor_time_vector=unique_censor_times,
                                           censor_weights=ipcw)

        # Computing the overall weight
        ipw = (iosw * iptw)[:, None] * ipcw
        return ipw.T


def ef_weight_models_censor(theta, s, a, c, sample_matrix, action_matrix, censor_matrix,
                            time_matrix_s1, final_time_matrix_s1, risk_set_matrix_s1, contribute_s1, strata_s1_times,
                            time_matrix_s0, final_time_matrix_s0, risk_set_matrix_s0, contribute_s0, strata_s0_times):
    m_index1 = sample_matrix.shape[1]
    m_index2 = m_index1 + action_matrix.shape[1]
    m_index3 = censor_matrix.shape[1] + len(strata_s1_times)
    beta = theta[:m_index1]
    gamma = theta[m_index1:m_index2]
    phi = theta[m_index2:]
    phi1 = phi[:m_index3]
    phi0 = phi[m_index3:]

    # Nuisance models
    ee_sample = ef_sample_logit(beta, s, sample_matrix)
    ee_action = ef_action_logit(gamma, s, a, action_matrix)
    ee_censor1 = ef_pooled_logit(theta=phi1, delta=c, baseline_matrix=censor_matrix,
                                 time_matrix=time_matrix_s1, final_time_matrix=final_time_matrix_s1,
                                 risk_set_matrix=risk_set_matrix_s1, contribute=contribute_s1)
    ee_censor0 = ef_pooled_logit(theta=phi0, delta=c, baseline_matrix=censor_matrix,
                                 time_matrix=time_matrix_s0, final_time_matrix=final_time_matrix_s0,
                                 risk_set_matrix=risk_set_matrix_s0, contribute=contribute_s0)
    return np.vstack([ee_sample, ee_action, ee_censor1, ee_censor0])


def construct_weights_censor(theta, s, a, sample_matrix, action_matrix, censor_matrix,
                             time_matrix_s1, strata_s1_times, time_matrix_s0, strata_s0_times,
                             unique_censor_times, unique_event_times):
    m_index1 = sample_matrix.shape[1]
    m_index2 = m_index1 + action_matrix.shape[1]
    m_index3 = censor_matrix.shape[1] + len(strata_s1_times)
    beta = theta[:m_index1]
    gamma = theta[m_index1:m_index2]
    phi = theta[m_index2:]
    phi1 = phi[:m_index3]
    phi0 = phi[m_index3:]

    # Constructing the inverse odds of sampling weights
    pr_s = inverse_logit(np.dot(sample_matrix, beta))
    iosw = s + ((1-s)*(1-pr_s)/pr_s)

    # Constructing the inverse probability of treatment weights
    gamma1 = gamma[:action_matrix.shape[1]]
    gamma0 = gamma[action_matrix.shape[1]:]
    pr_a2 = inverse_logit(np.dot(action_matrix, gamma1))
    pr_a1 = inverse_logit(np.dot(action_matrix, gamma0))
    iptw = ((a == 2) / pr_a2
            + (a == 1)*(s == 1) / (1-pr_a2)
            + (a == 1)*(s == 0) / pr_a1
            + (a == 0) / (1-pr_a1))

    # Constructing the inverse probability of censoring weights
    ipcw_s1 = 1 - plogit_predictions(theta=phi1, baseline_matrix=censor_matrix, time_matrix=time_matrix_s1,
                                     strata_unique_times=strata_s1_times, all_unique_times=unique_censor_times)
    ipcw_s0 = 1 - plogit_predictions(theta=phi0, baseline_matrix=censor_matrix, time_matrix=time_matrix_s0,
                                     strata_unique_times=strata_s0_times, all_unique_times=unique_censor_times)
    ipcw = ipcw_s1*s + ipcw_s0*(1-s)

    # Aligning censor weights
    ipcw = align_ipcw_with_event_times(event_time_vector=unique_event_times, censor_time_vector=unique_censor_times,
                                       censor_weights=ipcw)

    # Computing the overall weight
    ipw = iosw * iptw * ipcw
    return ipw


def ef_product_limit(theta, delta, final_time_matrix, risk_set_matrix, contributions, weights=None):
    delta = np.asarray(delta)                            # Convert delta to NumPy array
    theta = np.asarray(theta)                            # Convert theta to NumPy array

    # Preparing the theta vectors for the vectorized estimating equation
    theta = 1 - theta                                    # ... theta = S(t) = 1 - F(t) is applied
    theta_init = 1                                       # Survival at t=0 is defined as 1
    theta_s = np.insert(theta, 0, theta_init)[:-1]       # Shifting to get previous theta

    # Vectorized Kaplan-Meier functionality
    event_matrix = final_time_matrix * delta
    ee_risk = (risk_set_matrix.T * (theta - theta_s + theta_s*event_matrix.T)).T
    if weights is not None:
        ee_risk = ee_risk * weights

    # Returning the stacked estimating equations
    return ee_risk * contributions


def product_limit_predict(delta, final_time_matrix, risk_set_matrix, contributions, weights=None):
    if weights is None:
        weights = 1

    numer_matrix = final_time_matrix * delta * weights * contributions
    denom_matrix = risk_set_matrix * weights * contributions
    numerator = np.sum(numer_matrix, axis=1)
    denominator = np.sum(denom_matrix, axis=1)
    return np.cumprod(1 - (numerator / denominator))


def ef_risk_product_limit(theta, s, a, c, delta, sample_matrix, action_matrix, censor_matrix,
                          time_matrix_s1, final_time_matrix_s1, risk_set_matrix_s1, strata_s1_times,
                          time_matrix_s0, final_time_matrix_s0, risk_set_matrix_s0, strata_s0_times,
                          final_time_matrix, risk_set_matrix, contribute,
                          unique_censor_times, unique_event_times):
    n_unique_times = risk_set_matrix.shape[0]
    alpha = theta[:n_unique_times]
    beta_all = theta[n_unique_times:]

    # Nuisance models for weights and getting back weights
    ee_weights = ef_weight_models(theta=beta_all, s=s, a=a, c=c,
                                  sample_matrix=sample_matrix,
                                  action_matrix=action_matrix,
                                  censor_matrix=censor_matrix,
                                  time_matrix_s1=time_matrix_s1,
                                  final_time_matrix_s1=final_time_matrix_s1,
                                  risk_set_matrix_s1=risk_set_matrix_s1,
                                  time_matrix_s0=time_matrix_s0,
                                  final_time_matrix_s0=final_time_matrix_s0,
                                  risk_set_matrix_s0=risk_set_matrix_s0)
    ipw = construct_weights(theta=beta_all, s=s, a=a,
                            sample_matrix=sample_matrix, action_matrix=action_matrix, censor_matrix=censor_matrix,
                            time_matrix_s1=time_matrix_s1, strata_s1_times=strata_s1_times,
                            time_matrix_s0=time_matrix_s0, strata_s0_times=strata_s0_times,
                            unique_censor_times=unique_censor_times,
                            unique_event_times=unique_event_times)

    # Estimating risk function via product limit
    ee_risk = ef_product_limit(theta=alpha,
                               delta=delta,
                               final_time_matrix=final_time_matrix,
                               risk_set_matrix=risk_set_matrix,
                               contributions=contribute,
                               weights=ipw)

    # Returning stacked estimating functions
    return np.vstack([ee_risk, ee_weights])


def ef_diagnostic_product_limit(theta, s, a, c,
                                delta, sample_matrix, action_matrix, censor_matrix,
                                time_matrix_s1, final_time_matrix_s1, risk_set_matrix_s1, strata_s1_times,
                                time_matrix_s0, final_time_matrix_s0, risk_set_matrix_s0, strata_s0_times,
                                final_time_matrix, risk_set_matrix, unique_event_times, unique_censor_times):
    n_unique_times = risk_set_matrix.shape[0]
    obs_n = delta.shape[0]
    ird = theta[0]
    risk_diagnostic = np.asarray(theta[1: 1+n_unique_times])
    risk1 = np.asarray(theta[1+n_unique_times: 1+n_unique_times*2])
    risk0 = np.asarray(theta[1+n_unique_times*2: 1+n_unique_times*3])
    beta_all = theta[1+n_unique_times*3:]

    # Nuisance models for weights and getting back weights
    ee_weights = ef_weight_models(theta=beta_all, s=s, a=a, c=c,
                                  sample_matrix=sample_matrix, action_matrix=action_matrix,
                                  censor_matrix=censor_matrix,
                                  time_matrix_s1=time_matrix_s1,
                                  final_time_matrix_s1=final_time_matrix_s1,
                                  risk_set_matrix_s1=risk_set_matrix_s1,
                                  time_matrix_s0=time_matrix_s0,
                                  final_time_matrix_s0=final_time_matrix_s0,
                                  risk_set_matrix_s0=risk_set_matrix_s0)
    ipw = construct_weights(theta=beta_all, s=s, a=a,
                            sample_matrix=sample_matrix, action_matrix=action_matrix, censor_matrix=censor_matrix,
                            time_matrix_s1=time_matrix_s1, strata_s1_times=strata_s1_times,
                            time_matrix_s0=time_matrix_s0, strata_s0_times=strata_s0_times,
                            unique_censor_times=unique_censor_times,
                            unique_event_times=unique_event_times
                            )

    # Estimating risk function via product limit
    ee_risk1 = ef_product_limit(theta=risk1,
                                delta=delta,
                                final_time_matrix=final_time_matrix,
                                risk_set_matrix=risk_set_matrix,
                                contributions=(s == 1) & (a == 1),
                                weights=ipw)
    ee_risk0 = ef_product_limit(theta=risk0,
                                delta=delta,
                                final_time_matrix=final_time_matrix,
                                risk_set_matrix=risk_set_matrix,
                                contributions=(s == 0) & (a == 1),
                                weights=ipw)

    # Diagnostic risk function estimation
    ee_diag = ((risk1 - risk0) - np.asarray(risk_diagnostic))[:, None] * np.ones(obs_n)
    # Integrated risk difference
    unique_times = np.asarray(unique_event_times)
    time_scale = np.append(unique_times[1:], unique_times[-1]) - unique_times
    ee_ird = (ird - np.sum(risk_diagnostic * time_scale)) * np.ones(obs_n)
    # Returning stacked estimating equations
    return np.vstack([ee_ird, ee_diag, ee_risk1, ee_risk0, ee_weights])


def ef_pooled_logit(theta, delta, baseline_matrix, time_matrix, final_time_matrix, risk_set_matrix, contribute):
    baseline_n_params = baseline_matrix.shape[1]
    beta_x = theta[:baseline_n_params]
    beta_s = np.asarray(theta[baseline_n_params:])
    n_time_steps = time_matrix.shape[1]
    n_ones = np.ones(shape=(1, n_time_steps))

    # Log-odds contributions for covariate and time
    log_odds_w = np.dot(baseline_matrix, beta_x)
    log_odds_t = np.dot(time_matrix, beta_s)

    # Computing residuals
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))
    y_obs = delta * final_time_matrix
    y_pred = inverse_logit(log_odds_w_matrix + log_odds_t[:, None])
    residual_matrix = (y_obs - y_pred) * risk_set_matrix

    # Getting score matrix for X and S
    y_resid = np.dot(n_ones, residual_matrix)[0]
    x_score = y_resid[:, None] * baseline_matrix
    t_score = residual_matrix
    score_plogit = np.vstack([x_score.T, t_score])

    # Returning the score function for the stacked sums
    return score_plogit * contribute


def plogit_predictions(theta, baseline_matrix, time_matrix, strata_unique_times, all_unique_times):
    baseline_n_params = baseline_matrix.shape[1]
    obs_n = baseline_matrix.shape[0]
    beta_x = theta[:baseline_n_params]
    beta_s = np.asarray(theta[baseline_n_params:])
    n_time_steps = time_matrix.shape[1]

    # Log-odds contributions for covariate and time
    log_odds_w = np.dot(baseline_matrix, beta_x)
    log_odds_t = np.dot(time_matrix, beta_s)

    # # Computing full matrix of predicted values for each time
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))       # Stacked copies of X contributions for intervals
    y_pred = inverse_logit(log_odds_w_matrix + log_odds_t[:, None])  # Predicted event at time intervals matrix
    survival_prediction = np.cumprod(1 - y_pred, axis=0)
    risk_matrix = 1 - survival_prediction
    risk_t0 = 0

    # Generating predictions
    predictions = []
    for aut in all_unique_times:
        if aut == 0 or aut < strata_unique_times[0]:
            prediction = np.ones(obs_n) * risk_t0
        else:
            if strata_unique_times[-1] <= aut:
                matrix_index = -1
            else:
                further_times = strata_unique_times[aut < strata_unique_times]
                if len(further_times) < 1:
                    nearest = strata_unique_times[aut >= strata_unique_times][-1]
                else:
                    nearest = strata_unique_times[aut < strata_unique_times][0]
                matrix_index = np.where(strata_unique_times == nearest)[0][0] - 1
            prediction = risk_matrix[matrix_index, :]
        predictions.append(prediction)
    return np.asarray(predictions)


def ef_risk_function(theta, delta, baseline_matrix, time_matrix, final_time_matrix, risk_set_matrix, contribute,
                     strata_unique_times, all_unique_times):
    n_unique_times = len(all_unique_times)
    alpha = theta[:n_unique_times]
    beta = theta[n_unique_times:]
    # Pooled logistic estimation
    ee_plogit = ef_pooled_logit(theta=beta,
                                delta=delta,
                                baseline_matrix=baseline_matrix,
                                time_matrix=time_matrix,
                                final_time_matrix=final_time_matrix,
                                risk_set_matrix=risk_set_matrix,
                                contribute=contribute)
    # Risk function estimation
    risks = plogit_predictions(theta=beta,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix,
                               strata_unique_times=strata_unique_times,
                               all_unique_times=all_unique_times)
    ee_risks = risks - np.asarray(alpha)[:, None]
    # Returning stacked estimating equations
    return np.vstack([ee_risks, ee_plogit])


def ef_diagnostic_function(theta, delta, baseline_matrix, sample,
                           time_matrix_s1, final_time_matrix_s1, risk_set_matrix_s1, contribute_s1, strata_s1_times,
                           time_matrix_s0, final_time_matrix_s0, risk_set_matrix_s0, contribute_s0, strata_s0_times,
                           all_unique_times):
    n_unique_times = len(all_unique_times)
    obs_n = delta.shape[0]
    plogit_index = 1 + n_unique_times + baseline_matrix.shape[1] + len(strata_s1_times)
    ird = theta[0]
    alpha = theta[1:n_unique_times+1]
    beta1 = theta[n_unique_times+1:plogit_index]
    beta0 = theta[plogit_index:]
    # Pooled logistic estimation
    ee_plogit1 = ef_pooled_logit(theta=beta1,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s1,
                                 final_time_matrix=final_time_matrix_s1,
                                 risk_set_matrix=risk_set_matrix_s1,
                                 contribute=contribute_s1)
    ee_plogit0 = ef_pooled_logit(theta=beta0,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s0,
                                 final_time_matrix=final_time_matrix_s0,
                                 risk_set_matrix=risk_set_matrix_s0,
                                 contribute=contribute_s0)
    # Risk function estimation
    risk1 = plogit_predictions(theta=beta1,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s1,
                               strata_unique_times=strata_s1_times,
                               all_unique_times=all_unique_times)
    risk0 = plogit_predictions(theta=beta0,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s0,
                               strata_unique_times=strata_s0_times,
                               all_unique_times=all_unique_times)
    ee_diag = sample*(risk1 - risk0) - np.asarray(alpha)[:, None]
    # Integrated risk difference
    unique_times = np.asarray(all_unique_times)
    time_scale = np.append(unique_times[1:], unique_times[-1]) - unique_times
    ee_ird = (ird - np.sum(alpha * time_scale)) * np.ones(obs_n)
    # Returning stacked estimating equations
    return np.vstack([ee_ird, ee_diag, ee_plogit1, ee_plogit0])


def ef_single_span_function(theta, delta, baseline_matrix, sample,
                            time_matrix_s1, final_time_matrix_s1, risk_set_matrix_s1, contribute_s1, strata_s1_times,
                            time_matrix_s0, final_time_matrix_s0, risk_set_matrix_s0, contribute_s0, strata_s0_times,
                            all_unique_times):
    n_unique_times = len(all_unique_times)
    plogit_index = n_unique_times + baseline_matrix.shape[1] + len(strata_s1_times)
    alpha = theta[:n_unique_times]
    beta1 = theta[n_unique_times:plogit_index]
    beta0 = theta[plogit_index:]
    # Pooled logistic estimation
    ee_plogit1 = ef_pooled_logit(theta=beta1,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s1,
                                 final_time_matrix=final_time_matrix_s1,
                                 risk_set_matrix=risk_set_matrix_s1,
                                 contribute=contribute_s1)
    ee_plogit0 = ef_pooled_logit(theta=beta0,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s0,
                                 final_time_matrix=final_time_matrix_s0,
                                 risk_set_matrix=risk_set_matrix_s0,
                                 contribute=contribute_s0)
    # Risk function estimation
    risk1 = plogit_predictions(theta=beta1,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s1,
                               strata_unique_times=strata_s1_times,
                               all_unique_times=all_unique_times)
    risk0 = plogit_predictions(theta=beta0,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s0,
                               strata_unique_times=strata_s0_times,
                               all_unique_times=all_unique_times)
    ee_rd = sample*(risk1 - risk0) - np.asarray(alpha)[:, None]
    # Returning stacked estimating equations
    return np.vstack([ee_rd, ee_plogit1, ee_plogit0])


def ef_multi_span_function(theta, delta, baseline_matrix, sample,
                           time_matrix_s3, final_time_matrix_s3, risk_set_matrix_s3, contribute_s3, strata_s3_times,
                           time_matrix_s2, final_time_matrix_s2, risk_set_matrix_s2, contribute_s2, strata_s2_times,
                           time_matrix_s1, final_time_matrix_s1, risk_set_matrix_s1, contribute_s1, strata_s1_times,
                           time_matrix_s0, final_time_matrix_s0, risk_set_matrix_s0, contribute_s0, strata_s0_times,
                           all_unique_times):
    n_unique_times = len(all_unique_times)
    plogit_index3 = n_unique_times + baseline_matrix.shape[1] + len(strata_s3_times)
    plogit_index2 = plogit_index3 + baseline_matrix.shape[1] + len(strata_s2_times)
    plogit_index1 = plogit_index2 + baseline_matrix.shape[1] + len(strata_s1_times)

    alpha = theta[:n_unique_times]
    beta3 = theta[n_unique_times:plogit_index3]
    beta2 = theta[plogit_index3:plogit_index2]
    beta1 = theta[plogit_index2:plogit_index1]
    beta0 = theta[plogit_index1:]
    # Pooled logistic estimation
    ee_plogit3 = ef_pooled_logit(theta=beta3,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s3,
                                 final_time_matrix=final_time_matrix_s3,
                                 risk_set_matrix=risk_set_matrix_s3,
                                 contribute=contribute_s3)
    ee_plogit2 = ef_pooled_logit(theta=beta2,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s2,
                                 final_time_matrix=final_time_matrix_s2,
                                 risk_set_matrix=risk_set_matrix_s2,
                                 contribute=contribute_s2)
    ee_plogit1 = ef_pooled_logit(theta=beta1,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s1,
                                 final_time_matrix=final_time_matrix_s1,
                                 risk_set_matrix=risk_set_matrix_s1,
                                 contribute=contribute_s1)
    ee_plogit0 = ef_pooled_logit(theta=beta0,
                                 delta=delta,
                                 baseline_matrix=baseline_matrix,
                                 time_matrix=time_matrix_s0,
                                 final_time_matrix=final_time_matrix_s0,
                                 risk_set_matrix=risk_set_matrix_s0,
                                 contribute=contribute_s0)
    # Risk function estimation
    risk3 = plogit_predictions(theta=beta3,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s3,
                               strata_unique_times=strata_s3_times,
                               all_unique_times=all_unique_times)
    risk2 = plogit_predictions(theta=beta2,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s2,
                               strata_unique_times=strata_s2_times,
                               all_unique_times=all_unique_times)
    risk1 = plogit_predictions(theta=beta1,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s1,
                               strata_unique_times=strata_s1_times,
                               all_unique_times=all_unique_times)
    risk0 = plogit_predictions(theta=beta0,
                               baseline_matrix=baseline_matrix,
                               time_matrix=time_matrix_s0,
                               strata_unique_times=strata_s0_times,
                               all_unique_times=all_unique_times)
    ee_rd = sample*((risk3 - risk2) + (risk1 - risk0)) - np.asarray(alpha)[:, None]
    # Returning stacked estimating equations
    return np.vstack([ee_rd, ee_plogit3, ee_plogit2, ee_plogit1, ee_plogit0])
