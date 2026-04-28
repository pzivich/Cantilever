import warnings
import numpy as np
import pandas as pd
from delicatessen.utilities import identity, inverse_logit

from cantilever.formulas import get_design_matrix
from cantilever.estimators.point.efuncs import psi_action, psi_sample, psi_missing, psi_outcome
from cantilever.estimators.utils import (fit_mestimator,
                                         compute_action_score, compute_sample_score, compute_missing_score,
                                         descriptive_stats_w, descriptive_stats_r,
                                         print_nuisance_model_results)


class BridgePointEstimator:
    """Parent class for the bridged comparison point estimators.

    This class provides the skeleton for the bridged estimators for the mean. Basic functions implemented include:
    data processing, nuisance model estimators, background functionalities, diagnostic procedures, plotting.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame object consisting of all variables of interest
    outcome : str
        Column label for the outcome variable
    action : str
        Column label for the exposure variable
    sample : str
        Column label for the sample or study indicator variable
    alpha : float, optional
        Alpha level to use for the confidence intervals
    """
    def __init__(self, data, outcome, action, sample, alpha=0.05, verbose=True, decimals=2):
        # Checking if there is any missing data that occurs (besides missing outcome data)
        valid_obs = data.dropna(subset=[d for d in data.columns if d != outcome]    # Dropping all missing but outcome
                                ).shape[0]                                          # ... then getting the N
        original_size = data.shape[0]                                               # Getting the original N
        if valid_obs != original_size:                                              # Tell the user dropping missing
            warnings.warn("There is missing data that is not the outcome in the "
                          "data set. The estimator will drop all missing data that "
                          "is not the missing outcome data by default. The estimator "
                          "will use " + str(valid_obs) + " of " + str(original_size) +
                          " observations.",
                          UserWarning)
            data = data.dropna(subset=[d for d in data.columns if d != outcome]     # Drop missing data
                               ).copy()                                             # ... and create a copy of data

        # Saving data and column references
        self.outcome = outcome                      # Save designated outcome column
        self.action = action                        # Save designated exposure column
        self.sample = sample                        # Save designated exposure column
        self.alpha = alpha                          # Save designated alpha for CI
        self.data = data.copy()                     # Save copy of the data (copy regardless of previous)
        # TODO sort by action and sample

        # Checking if A is appropriately formatted
        # TODO check if A is 3 levels appropriately

        # Checking if S is appropriately formatted
        # TODO check if S has 2 levels appropriately

        # Checking if there is missing outcome data (so it can be handled internally)
        self.missing = "_missing_y_var_"                                # New column label for all missing data
        if valid_obs != data.dropna(subset=[outcome]).shape[0]:         # If missing outcomes occur
            self.__missing_outcome_flag__ = True                        # ... sets flag for later steps
            self.data[self.missing] = np.where(data[outcome].isna(),
                                               0, 1)
        else:                                                           # If no missing outcomes
            self.__missing_outcome_flag__ = False                       # ... sets flag as off (to ignore some warnings)
            self.data[self.missing] = 1                                 # ... mark all as observed

        # Checking if Y is continuous
        if data[outcome].value_counts().index.isin([0, 1]).all():       # If all Y values are 0 or 1
            self.__continuous_outcome_flag__ = False                    # ... set as not continuous outcome
            self.__outcome_type__ = "Binary"                            # ... display for summary
            self._outcome_model_dist_ = 'none'                          # ... distribution for outcome model
        else:                                                           # Otherwise
            self.__continuous_outcome_flag__ = True                     # ... mark as continuous outcome
            self.__outcome_type__ = "Continuous"                        # ... display for summary
            self._outcome_model_dist_ = 'none'                          # ... distribution for outcome model

        # Saving some meta information about data processing
        self.__original_n__ = original_size                             # Save the original N of input data
        self.__estimator_n__ = valid_obs                                # Save N after dropping missing (besides Y)
        self.__outcomes_n__ = np.sum(self.data[self.missing])           # Save N with Y measured
        self.__estimator_label__ = " Base-class"                        # Save name of estimator (updated later)
        self._verbose_ = verbose
        self._decimals_ = decimals

        # Defining some meta-parameters for the nuisance model fitting process
        self._action_nuisance_model_ = None                             # Storage for model specification for A
        self._action_nuisance_estimator_ = None                         # Storage for estimator of nuisance A
        self._action_design_matrix_ = None                              # Storage for design matrix for nuisance A
        self._action_coefs_ = None                                      # Storage for coefficients for A
        self._action_coefs_labels_ = None                               # Storage for coefficients labels for A
        self._sample_nuisance_model_ = None                             # Storage for model specification for S
        self._sample_nuisance_estimator_ = None                         # Storage for estimator of nuisance S
        self._sample_design_matrix_ = None                              # Storage for design matrix for nuisance S
        self._sample_coefs_ = None                                      # Storage for coefficients for S
        self._sample_coefs_labels_ = None                               # Storage for coefficients labels for S
        self._missing_nuisance_model_ = None                            # Storage for model specification for R
        self._missing_nuisance_estimator_ = None                        # Storage for estimator of nuisance R
        self._missing_design_matrix_ = None                             # Storage for design matrix for nuisance R
        self._missing_coefs_ = None                                     # Storage for coefficients for R
        self._missing_coefs_labels_ = None                              # Storage for coefficients labels for R
        self._outcome_nuisance_model_ = None                            # Storage for model specification for Y
        self._outcome_nuisance_estimator_ = None                        # Storage for estimator of nuisance Y
        self._outcome_design_matrix_ = None                             # Storage for design matrix for nuisance Y
        self._outcome_coefs_ = None                                     # Storage for coefficients for Y
        self._outcome_coefs_labels_ = None                              # Storage for coefficients labels for Y

        # Defining additional meta-parameters for estimation procedures
        self._truncation_pract_ = None                                  # Storage for output meta on action truncation
        self._truncation_prsamp_ = None                                 # Storage for output meta on sample truncation
        self._truncation_prmiss_ = None                                 # Storage for output meta on missing truncation
        self._aipw_implementation_ = None                               # Storage for AIPW implementation method

        # M-estimator procedure details
        self.solver = 'lm'                 # Default solving method for root-finding
        self.maxiter = 5000                # Default maximum iterations for root-finding
        self.tolerance = 1e-9              # Default tolerance for root-finding
        self.deriv_method = 'approx'       # Default differentiation method for sandwich
        self.dx = 1e-9                     # Default approximation distance for sandwich

        # Results
        self.mestimator = None             # Storage for completed M-estimator
        self.tmle_results = None           # Storage for completed TMLE process

    def action_model(self, model, init=None, bounds=(0, 1)):
        r"""Estimate the action nuisance model. This function takes the specified nuisance model for the action (e.g.,
        exposure, treatment, intervention) and estimates the corresponding parameters using a logistic regression model.
        This model is used to compute the propensity scores and inverse probability of treatment weights.

        Note
        ----
        While one does not need to include covariates in this model with a marginally randomized trial, including strong
        predictors of the outcome in this model may improve efficiency (i.e., produce narrower confidence intervals).


        The propensity score is defined as :math:`\Pr(A=a|W,S)`. Here, the action nuisance model is independently
        fit for each study sample.

        Parameters
        ----------
        model
        init
        bounds

        Returns
        -------

        """
        # Setting up data for the estimating functions
        y_nan, a, s, m = self._get_variable_arrays_()
        dm, labels = get_design_matrix(formula=model, data=self.data)
        n_params = dm.shape[1]

        # Defining the estimating functions
        def psi(theta):
            return psi_action(theta=theta, Z=dm, a=a, s=s)

        # Solving the estimating equations
        init = self._generate_inits_(init=init, n_params=2*n_params)
        estr = self._fit_mestimator_(estimating_functions=psi, init=init)

        # Printing details to console based on verbose flag
        if self._verbose_:
            print("====================================================================")
            print("Action Nuisance Model")
            print("--------------------------------------------------------------------")
            self._print_nuisance_fit_details_(n_obs=self.__estimator_n__,
                                              dep_var=self.action,
                                              family="logistic")
            print("====================================================================")
            print_nuisance_model_results(labels=["S=1 : " + l for l in labels] + ["S=0 : " + l for l in labels],
                                         m_estimator=estr, decimals=self._decimals_)
            print("====================================================================")

        # Storing outcome model details
        self._action_nuisance_model_ = model             # Action nuisance model specification
        self._truncation_pract_ = bounds                 # Probability clip points for IPTW
        self._action_design_matrix_ = dm                 # Action design matrix from model specification
        self._action_coefs_labels_ = labels              # Action nuisance model coefficient labels
        self._action_coefs_ = estr.theta                 # Action nuisance model coefficients

    def sample_model(self, model, init=None, bounds=(0, 1)):
        r"""Estimate the sample nuisance model. This function takes the specified nuisance model for sampling and
        estimates the corresponding parameters using a logistic regression model. This model is used to compute the
        sample scores and inverse odds of sampling weights.

        The sample score is defined as :math:`\Pr(S=1|W)`


        Parameters
        ----------
        model
        init
        bounds

        Returns
        -------

        """
        # Setting up data for the estimating functions
        y_nan, a, s, m = self._get_variable_arrays_()
        dm, labels = get_design_matrix(formula=model, data=self.data)
        n_params = dm.shape[1]

        # Defining the estimating functions
        def psi(theta):
            return psi_sample(theta=theta, V=dm, s=s)

        # Solving the estimating equations
        init = self._generate_inits_(init=init, n_params=n_params)
        estr = self._fit_mestimator_(estimating_functions=psi, init=init)

        # Printing details to console based on verbose flag
        if self._verbose_:
            print("====================================================================")
            print("Sampling Nuisance Model")
            print("--------------------------------------------------------------------")
            self._print_nuisance_fit_details_(n_obs=self.__estimator_n__,
                                              dep_var=self.sample,
                                              family="logistic")
            print("====================================================================")
            print_nuisance_model_results(labels=labels, m_estimator=estr, decimals=self._decimals_)
            print("====================================================================")

        # Storing outcome model details
        self._sample_nuisance_model_ = model             # Sampling nuisance model specification
        self._truncation_prsamp_ = bounds                # Probability clip points for IOSW
        self._sample_design_matrix_ = dm                 # Sampling design matrix from model specification
        self._sample_coefs_labels_ = labels              # Sampling nuisance model coefficient labels
        self._sample_coefs_ = estr.theta                 # Sampling nuisance model coefficients

    def missing_model(self, model, init=None, bounds=(0, 1)):
        r"""Estimate the sample nuisance model. This function takes the specified nuisance model for sampling and
        estimates the corresponding parameters using a logistic regression model. This model is used to compute the
        sample scores and inverse odds of sampling weights.

        The missing score is defined as :math:`\Pr(R=1|A,W,S)`

        Parameters
        ----------
        model
        init
        bounds

        Returns
        -------

        """
        if self.__missing_outcome_flag__:
            # Setting up data for the estimating functions
            y_nan, a, s, m = self._get_variable_arrays_()
            dm, labels = get_design_matrix(formula=model, data=self.data)
            n_params = dm.shape[1]

            # Defining the estimating functions
            def psi(theta):
                return psi_missing(theta, W=dm, m=m)

            # Solving the estimating equations
            init = self._generate_inits_(init=init, n_params=n_params)
            estr = self._fit_mestimator_(estimating_functions=psi, init=init)

            # Printing details to console based on verbose flag
            if self._verbose_:
                print("====================================================================")
                print("Missingness Nuisance Model")
                print("--------------------------------------------------------------------")
                self._print_nuisance_fit_details_(n_obs=self.__estimator_n__,
                                                  dep_var=self.missing,
                                                  family="logistic")
                print("====================================================================")
                print_nuisance_model_results(labels=labels, m_estimator=estr, decimals=self._decimals_)
                print("====================================================================")

            # Storing outcome model details
            self._missing_nuisance_model_ = model        # Missing nuisance model specification
            self._truncation_prmiss_ = bounds            # Probability clip points for IPMW
            self._missing_design_matrix_ = dm            # Missing design matrix from model specification
            self._missing_coefs_labels_ = labels         # Missing nuisance model coefficient labels
            self._missing_coefs_ = estr.theta            # Missing nuisance model coefficients
        else:
            warnings.warn("No missing outcome data was detected. Therefore, a missing model is not needed. This step "
                          "is being skipped.")

    def outcome_model(self, model, model_type='linear', init=None):
        r"""Outcome models with the same specifications are fit to either piece

        The outcome model is defined as :math:`E(Y|A,W,S,R=1)`. Here, the outcome nuisance model is independently
        fit for each study sample.

        Parameters
        ----------
        model : str
            ...
        model_type : str, optional
            ...
        init : None, ndarray, list
            ...

        Returns
        -------

        """
        # Checking some parameters for the model fitting process
        if self.__continuous_outcome_flag__:
            self._outcome_model_dist_ = model_type
        else:
            self._outcome_model_dist_ = 'logistic'

        # Setting up data for the estimating functions
        y_nan, a, s, m = self._get_variable_arrays_()
        dm, labels = get_design_matrix(formula=model, data=self.data)
        n_params = dm.shape[1]

        # Defining the estimating functions
        def psi(theta):
            return psi_outcome(theta=theta, X=dm, y=y_nan, s=s, m=m,
                               model=self._outcome_model_dist_, weights=None)

        # Solving the estimating equations
        init = self._generate_inits_(init=init, n_params=2*n_params)
        estr = self._fit_mestimator_(estimating_functions=psi, init=init)

        # Printing details to console based on verbose flag
        if self._verbose_:
            print("====================================================================")
            print("Outcome Nuisance Model")
            print("--------------------------------------------------------------------")
            self._print_nuisance_fit_details_(n_obs=self.__outcomes_n__,
                                              dep_var=self.outcome,
                                              family=self._outcome_model_dist_)
            print("====================================================================")
            print_nuisance_model_results(labels=["S=1 : " + l for l in labels] + ["S=0 : " + l for l in labels],
                                         m_estimator=estr, decimals=self._decimals_)
            print("====================================================================")

        # Storing outcome model details
        self._outcome_nuisance_model_ = model          # Outcome nuisance model specification
        self._outcome_design_matrix_ = dm              # Outcome design matrix from model specification
        self._outcome_coefs_labels_ = labels           # Outcome nuisance model coefficient labels
        self._outcome_coefs_ = estr.theta              # Outcome nuisance model coefficients

    # TODO add weighted outcome model that AIPW only calls?

    def _get_variable_arrays_(self):
        """Internal function to

        Returns
        -------

        """
        y = np.asarray((self.data[self.outcome]))
        y_nan = np.nan_to_num(y, copy=True, nan=-9999.)
        a = np.asarray((self.data[self.action]))
        s = np.asarray((self.data[self.sample]))
        m = np.asarray((self.data[self.missing]))
        return y_nan, a, s, m

    def _get_updated_design_matrices_(self):
        """Internal function to compute the updated outcome model design matrices when the action is set to the
        different levels.

        Returns
        -------
        Three NumPy arrays, with A=2, A=1, A=0 respectively
        """
        da = self.data.copy()
        # I can trick patsy here by ensuring coverage of 0,1,2 with the S=0 group (may be dangerous)
        da[self.action] = np.where(da[self.sample] == 1, 2, da[self.action])
        Xa2, _ = get_design_matrix(formula=self._outcome_nuisance_model_, data=da)
        da[self.action] = np.where(da[self.sample] == 1, 1, da[self.action] * 2)
        Xa1, _ = get_design_matrix(formula=self._outcome_nuisance_model_, data=da)
        da[self.action] = np.where(da[self.sample] == 1, 0, da[self.action] + 1)
        Xa0, _ = get_design_matrix(formula=self._outcome_nuisance_model_, data=da)
        return Xa2, Xa1, Xa0

    @staticmethod
    def _generate_inits_(init, n_params):
        """Internal function to generate initial values

        Returns
        -------
        list
        """
        # TODO I should make inits with 'smart' intercepts
        if init is None:
            init = [0., ] * n_params
        else:
            if len(init) != n_params:
                raise ValueError("The length of the provided `init` does not match the number of parameters as "
                                 "determined by the estimating equations. There are " + str(n_params) + ", but "
                                 + str(len(init)) + " were given.")
        return list(init)

    def _fit_mestimator_(self, estimating_functions, init):
        """Internal function to fit the corresponding M-estimator

        Returns
        -------
        Optimized Delicatessen MEstimator class object
        """
        fmestr = fit_mestimator(estimating_functions, init,         # Apply M-estimator procedure
                                solver=self.solver,                 # ... what solver to use
                                maxiter=self.maxiter,               # ... number of iterations allowed
                                tolerance=self.tolerance,           # ... tolerance for the solution
                                deriv_method=self.deriv_method,     # ... derivative method to use
                                dx=self.dx,                         # ... derivative approximation space
                                subset=None)                        # ... never subset parameters
        return fmestr

    def results_table(self):
        """Generate a results table for the parameter of interest.

        Returns
        -------
        pandas DataFrame
        """
        if self.mestimator is None:
            raise ValueError("The estimation procedure must be completed before the results can be obtained. "
                             "Please check the order of the function calls in your code.")

        # Getting bridge estimator data
        est = self.mestimator.theta
        var = self.mestimator.variance
        ci = self.mestimator.confidence_intervals(alpha=self.alpha)
        pval = self.mestimator.p_values(null=0)

        # Formatting results into a dataframe
        columns = [" ", "Estimate", "SE", "LCL", "UCL", "P-value"]
        output = [["Single-span", est[0], var[0, 0]**0.5, ci[0, 0], ci[0, 1], pval[0]],
                  ["Multi-span", est[1], var[1, 1]**0.5, ci[1, 0], ci[1, 1], pval[1]],
                  ["Diagnostic", est[2], var[2, 2]**0.5, ci[2, 0], ci[2, 1], pval[2]],
                  ["A=2,S=1", est[3], var[3, 3]**0.5, ci[3, 0], ci[3, 1], np.nan],
                  ["A=1,S=1", est[4], var[4, 4]**0.5, ci[4, 0], ci[4, 1], np.nan],
                  ["A=1,S=0", est[5], var[5, 5]**0.5, ci[5, 0], ci[5, 1], np.nan],
                  ["A=0,S=0", est[6], var[6, 6]**0.5, ci[6, 0], ci[6, 1], np.nan],
                  ]
        res = pd.DataFrame(output, columns=columns)
        res = res.set_index(" ")
        return res

    def results_nuisance(self):
        if self.mestimator is None:
            raise ValueError("The estimation procedure must be completed before the results can be obtained. "
                             "Please check the order of the function calls in your code.")

        # Getting bridge estimator data
        est = self.mestimator.theta
        var = np.diag(self.mestimator.variance)
        ci = self.mestimator.confidence_intervals(alpha=self.alpha)
        pval = self.mestimator.p_values(null=0)
        id_n = 7

        # Setting up outputs
        nuisance_labels = []
        nuisance_tables = []

        def regression_table_from_mestimator(variables, estimate, standard_error, cis, pvalues, id_start, id_end):
            result = pd.DataFrame()
            result['Variable'] = variables
            result['Estimate'] = estimate[id_start:id_end]
            result['SE'] = standard_error[id_start:id_end] ** 0.5
            result['LCL'] = cis[id_start:id_end, 0]
            result['UCL'] = cis[id_start:id_end, 1]
            result['P-value'] = pvalues[id_start:id_end]
            return result.set_index("Variable")

        # Outcome nuisance model(s)
        if self._outcome_nuisance_model_ is None:
            id_o0 = id_n
        else:
            len_os = self._outcome_design_matrix_.shape[1]
            id_o1 = id_n + len_os
            id_o0 = id_o1 + len_os
            r_m1 = regression_table_from_mestimator(variables=self._outcome_coefs_labels_,
                                                    estimate=est, standard_error=var**0.5,
                                                    cis=ci, pvalues=pval,
                                                    id_start=id_n, id_end=id_o1)
            r_m0 = regression_table_from_mestimator(variables=self._outcome_coefs_labels_,
                                                    estimate=est, standard_error=var**0.5,
                                                    cis=ci, pvalues=pval,
                                                    id_start=id_o1, id_end=id_o0)
            nuisance_labels = nuisance_labels + ["Outcome model - S=1", "Outcome model - S=0"]
            nuisance_tables = nuisance_tables + [r_m1, r_m0]

        # Weighting nuisance model(s)
        if self._action_nuisance_model_ is not None:
            len_as = self._action_design_matrix_.shape[1]
            id_a1 = id_o0 + len_as
            id_a0 = id_a1 + len_as
            len_s = self._sample_design_matrix_.shape[1]
            id_s = id_a0 + len_s

            # Action nuisance models
            r_a1 = regression_table_from_mestimator(variables=self._action_coefs_labels_,
                                                    estimate=est, standard_error=var**0.5,
                                                    cis=ci, pvalues=pval,
                                                    id_start=id_o0, id_end=id_a1)
            r_a0 = regression_table_from_mestimator(variables=self._action_coefs_labels_,
                                                    estimate=est, standard_error=var**0.5,
                                                    cis=ci, pvalues=pval,
                                                    id_start=id_a1, id_end=id_a0)

            # Sampling nuisance model
            r_s = regression_table_from_mestimator(variables=self._sample_coefs_labels_,
                                                   estimate=est, standard_error=var**0.5,
                                                   cis=ci, pvalues=pval,
                                                   id_start=id_a0, id_end=id_s)
            nuisance_labels = nuisance_labels + ["Action model - S=1", "Action model - S=0", "Sampling Model"]
            nuisance_tables = nuisance_tables + [r_a1, r_a0, r_s]
            if self._missing_nuisance_model_ is not None:
                # Missing nuisance models
                len_m = self._missing_design_matrix_.shape[1]
                id_m = id_s + len_m
                r_m = regression_table_from_mestimator(variables=self._missing_coefs_labels_,
                                                       estimate=est, standard_error=var**0.5,
                                                       cis=ci, pvalues=pval,
                                                       id_start=id_s, id_end=id_m)
                nuisance_labels = nuisance_labels + ["Missing Model", ]
                nuisance_tables = nuisance_tables + [r_m, ]

        return nuisance_tables, nuisance_labels

    def _print_nuisance_fit_details_(self, n_obs, dep_var, family):
        """Internal function to describe nuisance model specifications

        Returns
        -------
        None
        """
        fmt = "No. Observations:   {:<11} | Dependent Variable: {:<11}"
        print(fmt.format(n_obs, dep_var))
        fmt = "Model:              {:<11} | Method:             {:<11}"
        print(fmt.format(family, self.solver))

    def _summary_details_(self):
        fmt = "No. Observations: {:<10} | No. Input:        {:<10}"
        print(fmt.format(self.__estimator_n__, self.__original_n__))
        fmt = "No. w/ Outcomes:  {:<10} | Outcome:          {:<10}"
        print(fmt.format(self.__outcomes_n__, self.outcome))
        fmt = "Action:           {:<10} | Sample:           {:<10}"
        print(fmt.format(self.action, self.sample))
        fmt = "Outcome type:     {:<10} | Model:            {:<10}"
        print(fmt.format(self.__outcome_type__, self._outcome_model_dist_))
        fmt = "Alpha:            {:<10} | "
        print(fmt.format(self.alpha))

    def summary(self):
        """Display summary results

        Returns
        -------
        None
        """
        table = self.results_table()
        print("==============================================================")
        print("Estimator:       ", self.__estimator_label__)
        print("--------------------------------------------------------------")
        self._summary_details_()
        print("--------------------------------------------------------------")
        print(table.round(decimals=self._decimals_))
        print("==============================================================")

    def save_results(self, file):
        """Save the results table as a CSV file

        Parameters
        ----------
        file : str
            File name (and path) to save the results to. Defaults to current sys path

        Returns
        -------
        None
        """
        table = self.results_table()
        table.to_csv(file=file+".csv")

    def diagnostics_weights(self):
        """

        Returns
        -------
        None
        """
        if self._sample_coefs_ is None or self._action_coefs_ is None:
            raise ValueError("Nuisance models for weights must be specified prior to running the diagnostics...")

        y_nan, a, s, m = self._get_variable_arrays_()
        print("==============================================================")
        print("Weight Diagnostics")
        print("==============================================================")

        # IPTW diagnostics
        print("Inverse Probability of Treatment Weights")
        print(" * The overall 'Mean' column should be near 2")
        print(" * Action-specific 'Sum' columns should be approximately equal")
        pi_a = compute_action_score(a=a, s=s, design_Z=self._action_design_matrix_, param=self._action_coefs_,
                                    clip=self._truncation_pract_)
        iptw = 1 / pi_a
        a_s0 = a[s == 0]
        w_s0 = iptw[s == 0]
        print("--------------------------------------------------------------")
        print(self.sample + " = 0")
        print("--------------------------------------------------------------")
        descriptive_stats_w(var=a_s0, values=[0, 1], w=w_s0, label=self.action,
                            include_overall=True, decimals=self._decimals_)
        a_s1 = a[s == 1]
        w_s1 = iptw[s == 1]
        print("--------------------------------------------------------------")
        print(self.sample + " = 1")
        print("--------------------------------------------------------------")
        descriptive_stats_w(var=a_s1, values=[1, 2], w=w_s1, label=self.action,
                            include_overall=True, decimals=self._decimals_)
        print("==============================================================")

        # IOSW diagnostics
        print("Inverse Odds of Sampling Weights")
        print(" * The 'Sum' columns should be approximately equal")
        pi_s = compute_sample_score(s=s, design_V=self._sample_design_matrix_, param=self._sample_coefs_,
                                    clip=self._truncation_prsamp_)
        iosw = 1 / pi_s
        print("--------------------------------------------------------------")
        descriptive_stats_w(var=s, values=[0, 1], w=iosw, label=self.sample,
                            decimals=self._decimals_)
        print("==============================================================")

        # IPMW diagnostics
        if self._missing_nuisance_model_ is not None:
            pi_m = compute_missing_score(param=self._missing_coefs_, design_W=self._missing_design_matrix_,
                                         clip=self._truncation_prmiss_)
            ipmw = 1 / pi_m
            m_s0 = m[s == 0]
            w_s0 = ipmw[s == 0]
            print("Inverse Probability of Missing Weights")
            print("--------------------------------------------------------------")
            print(self.sample + " = 0")
            print(" * 'Sum' of " + self.sample + " = 0 should be approximately " + str(np.sum(1-s)))
            print("--------------------------------------------------------------")
            descriptive_stats_w(var=m_s0, values=[1, ], w=w_s0, label="Observed",
                                decimals=self._decimals_)
            m_s1 = m[s == 1]
            w_s1 = ipmw[s == 1]
            print("--------------------------------------------------------------")
            print(self.sample + " = 1")
            print(" * 'Sum' of " + self.sample + " = 1 should be approximately " + str(np.sum(s)))
            print("--------------------------------------------------------------")
            descriptive_stats_w(var=m_s1, values=[1, ], w=w_s1, label="Observed",
                                decimals=self._decimals_)
            print("==============================================================")
        else:
            ipmw = 1

        # Overall Weight Diagnostics
        print("Overall Inverse Probability Weights")
        print(" * Check no extreme weights appear after multiplying together")
        print("--------------------------------------------------------------")
        ipw = iptw * iosw * ipmw
        s_m1 = s[m == 1]
        w_m1 = ipw[m == 1]
        descriptive_stats_w(var=s_m1, values=[0, 1], w=w_m1, label=self.sample,
                            decimals=self._decimals_)
        print("==============================================================")

        # TODO other diagnostics: boxplot of scores, standardized mean differences

    def diagnostics_outcome(self):
        """

        Returns
        -------
        None
        """
        if self._outcome_coefs_ is None:
            raise ValueError("Outcome nuisance model must be specified prior to running the diagnostics...")

        y_nan, a, s, m = self._get_variable_arrays_()
        X = self._outcome_design_matrix_
        beta1 = self._outcome_coefs_[:X.shape[1]]
        beta0 = self._outcome_coefs_[X.shape[1]:]

        if self._outcome_model_dist_ == 'linear':
            transform = identity
        elif self._outcome_model_dist_ == 'logistic':
            transform = inverse_logit
        elif self._outcome_model_dist_ == 'poisson':
            transform = np.exp
        else:
            raise ValueError("The outcome model specification " + str(self._outcome_model_dist_) + " is not supported.")

        print("==============================================================")
        print("Outcome Regression Residuals")
        print(" * Skewed or large residual may indicate misspecification")
        print("--------------------------------------------------------------")
        y_hat = transform(np.dot(X, beta1))*s + transform(np.dot(X, beta0))*(1-s)
        s1 = s[m == 1]
        resid = (y_hat - y_nan)[m == 1]
        descriptive_stats_r(study=s1, residual=resid, label=self.sample, decimals=3)
        print("==============================================================")

    def diagnostic_shared(self):
        """

        Returns
        -------

        """
        print("==============================================================")
        print("Shared Arm Diagnostic")
        print("==============================================================")
        self._summary_details_()
        print("--------------------------------------------------------------")
        table = self.results_table()
        table = table.iloc[[2, 4, 5]]
        print(table.round(decimals=self._decimals_))
        print("==============================================================")

