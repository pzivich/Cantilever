import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delicatessen import MEstimator

from cantilever.formulas import get_design_matrix
from cantilever.plotting import twister_plot
from cantilever.estimators.time_to_event.basics import BridgeTimeEstimator


class BridgeAIPW(BridgeTimeEstimator):
    r"""AIPW bridge algorithm

    """
    def __init__(self, data, time, delta, action, sample, alpha=0.05, verbose=True, decimals=2):
        # initialize the preceding class (allows for more arguments in init than BaseEstimator)
        super().__init__(data=data, time=time, delta=delta, action=action, sample=sample,
                         alpha=alpha, verbose=verbose, decimals=decimals)

        # Updating specific parameters for g-computation
        self.__estimator_label__ = "Augmented Inverse Probability Weighting"

    def estimate_risks(self):
        if self._sample_nuisance_model_ is None:
            raise ValueError("The function sample_model() must be called prior to estimating the risks")
        if self._action_nuisance_model_ is None:
            raise ValueError("The function action_model() must be called prior to estimating the risks")
        if self._outcome_nuisance_model_ is None:
            raise ValueError("The function outcome_model() must be called prior to estimating the risks")

