import numpy as np
from delicatessen.utilities import inverse_logit


def align_ipcw_with_event_times(event_time_vector, censor_time_vector, censor_weights):
    # Setup
    dt_vector = np.asarray(event_time_vector)
    ct_vector = np.asarray(censor_time_vector)
    censor_weights = np.asarray(censor_weights).T

    # Creating empty weight matrix
    ipcw = np.ones((censor_weights.shape[0], dt_vector.shape[0]))
    for i in range(len(dt_vector)):
        dt = dt_vector[i]
        if dt < ct_vector[0]:
            pass
        else:
            j = np.searchsorted(ct_vector, dt, side='left')
            if j > len(ct_vector):
                ipcw[:, i] = censor_weights[:, -1]
            else:
                ipcw[:, i] = censor_weights[:, j-1]
    return ipcw


def product_limit_predict(delta, final_time_matrix, risk_set_matrix, contributions, weights=None):
    if weights is None:
        weights = 1

    numer_matrix = final_time_matrix * delta * weights * contributions
    denom_matrix = risk_set_matrix * weights * contributions
    numerator = np.sum(numer_matrix, axis=1)
    denominator = np.sum(denom_matrix, axis=1)
    return np.cumprod(1 - (numerator / denominator))


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


