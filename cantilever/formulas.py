import numpy as np
import pandas as pd
from formulaic import model_matrix


def get_design_matrix(formula, data):
    design_matrix_pandas = model_matrix(formula, data)
    return np.asarray(design_matrix_pandas), list(design_matrix_pandas.columns)
