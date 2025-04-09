import numpy as np


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
