import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delicatessen import MEstimator
from delicatessen.utilities import logit
from delicatessen.estimating_equations import ee_regression

from cantilever.formulas import get_design_matrix
from cantilever.plotting import twister_plot
from cantilever.estimators.time_to_event.efuncs import (ef_sample_logit,
                                                        ef_action_logit,
                                                        ef_pooled_logit)
from cantilever.estimators.utils import (fit_mestimator,
                                         compute_action_score, compute_sample_score, compute_missing_score,
                                         descriptive_stats_w, descriptive_stats_r,
                                         print_nuisance_model_results)


class BridgeTimeEstimator:
    """Parent or base class for the bridged comparisons mean recipes.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame object consisting of all variables of interest
    time : str
        Column label for the time variable
    delta : str
        Column label for the event indicator variable
    action : str
        Column label for the exposure variable
    sample : str
        Column label for the sample or study indicator variable
    censor : str, None, optional
        Column label for the censoring indicator
    alpha : float, optional
        Alpha level to use for the confidence intervals
    verbose : bool, optional
        Whether to print intermediate results
    decimals : int, optional
        Number of decimal places to display in outputs
    """
    def __init__(self, data, time, delta, action, sample, censor=None, alpha=0.05, verbose=True, decimals=2):
        # Checking if there is any missing data that occurs (besides missing outcome data)
        valid_obs = data.dropna().shape[0]                                          # N removing missing data
        original_size = data.shape[0]                                               # Getting the original N
        if valid_obs != original_size:                                              # Tell the user dropping missing
            warnings.warn("There is missing data that is not the outcome in the "
                          "data set. The estimator will drop all missing data that "
                          "is not the missing outcome data by default. The estimator "
                          "will use " + str(valid_obs) + " of " + str(original_size) +
                          " observations.",
                          UserWarning)
            self.data = data.dropna().copy()                                        # Create copy without missing data
        else:
            self.data = data.copy()

        # Saving data and column references
        self.time = time                            # Save designated time column
        self.delta = delta                          # Save designated event indicator column
        self.action = action                        # Save designated exposure column
        self.sample = sample                        # Save designated sample column
        self.censor = censor                        # Save designated censor indicator column
        self.alpha = alpha                          # Save designated alpha for CI
        # TODO sort by action and sample (maybe not needed)

        # Checking if A is appropriately formatted
        # TODO check if A is 3 levels appropriately

        # Checking if S is appropriately formatted
        # TODO check if S has 2 levels appropriately

        # Checking if delta is binary
        if not self.data[self.delta].value_counts().index.isin([0, 1]).all():
            raise ValueError("The event indicator, `delta`, must only take on values of 0 or 1")

        # Sorting out unique event times and different study-arm combinations
        self.all_unique_times = list(np.unique(self.data.loc[self.data[self.delta] == 1, self.time]))
        self._combinations_ = [[2, 1], [1, 1], [1, 0], [0, 0]]

        # Saving some meta information about data processing
        self.__original_n__ = original_size                             # Save the original N of input data
        self.__estimator_n__ = valid_obs                                # Save N after dropping missing (besides Y)
        self.__estimator_label__ = "Base-class"                         # Save name of estimator (updated later)
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
        self._censor_nuisance_model_ = None                             # Storage for model specification for R
        self._censor_nuisance_estimator_ = None                         # Storage for estimator of nuisance R
        self._censor_design_matrix_ = None                              # Storage for design matrix for nuisance R
        self._censor_coefs_ = None                                      # Storage for coefficients for R
        self._censor_coefs_labels_ = None                               # Storage for coefficients labels for R
        self._outcome_nuisance_model_ = None                            # Storage for model specification for Y
        self._outcome_nuisance_estimator_ = None                        # Storage for estimator of nuisance Y
        self._outcome_design_matrix_ = None                             # Storage for design matrix for nuisance Y
        self._outcome_coefs_ = None                                     # Storage for coefficients for Y
        self._outcome_coefs_labels_ = None                              # Storage for coefficients labels for Y

        # Defining additional meta-parameters for estimation procedures
        self._truncation_pract_ = None                                  # Storage for output meta on action truncation
        self._truncation_prsamp_ = None                                 # Storage for output meta on sample truncation
        self._truncation_prcens_ = None                                 # Storage for output meta on missing truncation
        self._aipw_implementation_ = None                               # Storage for AIPW implementation method

        # M-estimator procedure details
        self.solver = 'lm'                 # Default solving method for root-finding
        self.maxiter = 20000               # Default maximum iterations for root-finding
        self.tolerance = 1e-9              # Default tolerance for root-finding
        self.deriv_method = 'approx'       # Default differentiation method for sandwich
        self.dx = 1e-9                     # Default approximation distance for sandwich

        # Results
        self.mestimator = None             # Storage for completed M-estimator
        self.tmle_results = None           # Storage for completed TMLE process

        # Results storage
        self.risks = None
        self.diagnostic = None
        self.diagnostic_test = None
        self.single_span = None
        self.multi_span = None

    def action_model(self, model, init=None, bounds=(0, 1)):
        self._action_nuisance_model_ = model
        t, delta, a, s, c = self._get_variable_arrays_()
        dm, labels = get_design_matrix(self._action_nuisance_model_, self.data)
        n_params = dm.shape[1]

        # Defining the estimating functions
        def psi(theta):
            return ef_action_logit(theta=theta, s=s, a=a,
                                   action_matrix=dm)

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
        self._action_coefs_ = list(estr.theta)           # Action nuisance model coefficients

    def sample_model(self, model, init=None, bounds=(0, 1)):
        # Setting up data for the estimating functions
        t, delta, a, s, c = self._get_variable_arrays_()
        dm, labels = get_design_matrix(formula=model, data=self.data)
        n_params = dm.shape[1]

        def psi(theta):
            return ef_sample_logit(theta=theta, s=s, sample_matrix=dm)

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
        self._sample_coefs_ = list(estr.theta)           # Sampling nuisance model coefficients

    def censor_model(self, model, init=None, bounds=(0, 1)):
        # Setting up data for the estimating functions
        model = model + " - 1"
        t, delta, a, s, c = self._get_variable_arrays_()
        dm, labels = get_design_matrix(model, self.data)
        dm = np.asarray(dm)
        n_bcovs = dm.shape[1]

        # Fitting a series of pooled logistic models
        self._censor_coefs_ = []
        for samp in [1, 0]:
            t_matrix, f_matrix, r_matrix, u_times = self._get_censor_matrices_(t=t, c=c, sample=samp)
            n_t_steps = t_matrix.shape[1]

            def psi(theta):
                return ef_pooled_logit(theta=theta,
                                       delta=c,
                                       baseline_matrix=dm,
                                       time_matrix=t_matrix,
                                       final_time_matrix=f_matrix,
                                       risk_set_matrix=r_matrix,
                                       contribute=(s == samp))

            # Solving the estimating equations
            init = self._generate_inits_plr_(n_covs=n_bcovs, event=self.censor, s=samp)
            estr = self._fit_mestimator_(psi, init=init)

            # Printing details to console based on verbose flag
            if self._verbose_:
                print("====================================================================")
                print("Censoring Nuisance Model: S=" + str(samp))
                print("--------------------------------------------------------------------")
                self._print_nuisance_fit_details_(n_obs=self.data.loc[self.data[self.sample] == samp].shape[0],
                                                  dep_var=self.censor,
                                                  family="plogit")
                print("--------------------------------------------------------------------")
                print("* Only coefficients for baseline covariates are shown")
                print("====================================================================")
                print_nuisance_model_results(labels=list(labels) + ["_", ]*n_t_steps,
                                             m_estimator=estr, decimals=self._decimals_,
                                             subset=n_bcovs)
                print("====================================================================")

            # Storing coefficients for later fitting
            self._censor_coefs_.append(list(estr.theta))

        # Storing censoring model details
        self._censor_nuisance_model_ = model      # Censor nuisance model specification
        self._censor_design_matrix_ = dm          # Censor design matrix from model specification
        self._censor_coefs_labels_ = labels       # Censor nuisance model coefficient labels

    def outcome_model(self, model):
        # Setting up data for the estimating functions
        model = model + " - 1"
        t, delta, a, s, c = self._get_variable_arrays_()
        dm, labels = get_design_matrix(model, self.data)
        dm = np.asarray(dm)
        n_bcovs = dm.shape[1]

        # Fitting a series of pooled logistic models
        self._outcome_coefs_ = []
        for act_samp_combo in self._combinations_:
            act = act_samp_combo[0]
            samp = act_samp_combo[1]

            t_matrix, f_matrix, r_matrix, u_times = self._get_time_matrices_(t=t, action=act, sample=samp)
            n_t_steps = t_matrix.shape[1]

            def psi(theta):
                return ef_pooled_logit(theta=theta,
                                       delta=delta,
                                       baseline_matrix=dm,
                                       time_matrix=t_matrix,
                                       final_time_matrix=f_matrix,
                                       risk_set_matrix=r_matrix,
                                       contribute=(a == act) & (s == samp))

            # Solving the estimating equations
            init = self._generate_inits_plr_(n_covs=n_bcovs, event=self.delta, s=samp, a=act)
            estr = self._fit_mestimator_(psi, init=init)

            # Printing details to console based on verbose flag
            if self._verbose_:
                print("====================================================================")
                print("Outcome Nuisance Model: S=" + str(samp) + ", A="+str(act))
                print("--------------------------------------------------------------------")
                n_size = self.data.loc[(self.data[self.sample] == samp) & (self.data[self.action] == act)].shape[0]
                self._print_nuisance_fit_details_(n_obs=n_size, dep_var=self.delta, family="plogit")
                print("--------------------------------------------------------------------")
                print("* Only coefficients for baseline covariates are shown")
                print("====================================================================")
                print_nuisance_model_results(labels=list(labels) + ["_", ] * n_t_steps,
                                             m_estimator=estr, decimals=self._decimals_,
                                             subset=n_bcovs)
                print("====================================================================")

            # Storing coefficients for later fitting
            self._outcome_coefs_.append(list(estr.theta))

        # Storing censoring model details
        self._outcome_nuisance_model_ = model      # Censor nuisance model specification
        self._outcome_design_matrix_ = dm          # Censor design matrix from model specification
        self._outcome_coefs_labels_ = labels       # Censor nuisance model coefficient labels

    def estimate_risks(self):
        pass

    def estimate_diagnostic(self):
        pass

    def estimate_single_span(self):
        pass

    def estimate_multi_span(self):
        pass

    def estimate_contrasts(self):
        self.estimate_diagnostic()
        self.estimate_single_span()
        self.estimate_multi_span()

    def plot_risks(self, ax=None, colors=('c', 'm', 'y', 'k'), linestyle=('-', '-', '-', '-'),
                   labels=('A2S1', 'A1S1', 'A1S0', 'A0S0'), include_ci=False):
        if ax is None:                   # If not axes are provided
            ax = plt.gca()               # ... then generate a new axes to plot with

        # Estimating risks and their CI's
        if self.risks is None:
            self.estimate_risks()
        r = self.risks

        # Generating the plot
        for i in range(len(self._combinations_)):
            act_samp_combo = self._combinations_[i]
            act = act_samp_combo[0]
            samp = act_samp_combo[1]
            label = 'R-A' + str(act) + 'S' + str(samp)
            ax.step(r.index,                      # Step function with steps at time (x-axis)
                    r[label],                     # ... and the measure (y-axis)
                    where='post',                 # ... with steps at the post point
                    label=labels[i],              # ... and drop line from appearing in legend
                    color=colors[i],              # ... requested color
                    linestyle=linestyle[i])       # ... and the requested linestyle
            if include_ci:
                ax.fill_between(r.index,          # Shade in confidence interval across time
                                r[label+"_UCL"],  # ... from upper confidence limit
                                r[label+"_LCL"],  # ... to lower confidence limit
                                color=colors[i],  # ... as the requested color
                                alpha=0.2,        # ... with some transparency for ease
                                step='post')      # ... then apply steps as post

        ax.set_xlim([0, self.all_unique_times[-1] + 1])  # Setting x-axis limits based on time
        ax.set_xlabel("Time")  # Generic time label for x-axis
        ax.set_ylim([0, 1])
        ax.set_ylabel("Risk")  # Generic measure label for y-axis

        # Returning the plot to user
        return ax

    def plot_diagnostic(self, ax=None, color='k', favors=True, favors_label=("S=1", "S=0"), favors_spacing="\t"):
        # Estimating risks and their CI's1
        if self.diagnostic is None:
            self.estimate_diagnostic()
        r = self.diagnostic.reset_index()

        # Drawing plot
        ax = twister_plot(data=r, point='RD-D', lcl='RD-D_LCL', ucl='RD-D_UCL', time='Time',
                          color=color,
                          reference_line=0.0,
                          log_scale=False,
                          favors=favors,
                          favors_label=favors_label,
                          favors_spacing=favors_spacing,
                          step=True,
                          ax=ax)

        # Returning the plot to user
        return ax

    def plot_single_span(self, ax=None, color='k', favors=True, favors_label=("A=2", "A=0"), favors_spacing="\t"):
        # Estimating risks and their CI's
        if self.single_span is None:
            self.estimate_single_span()
        r = self.single_span.reset_index()

        # Drawing plot
        ax = twister_plot(data=r, point='RD-SS', lcl='RD-SS_LCL', ucl='RD-SS_UCL', time='Time',
                          color=color,
                          reference_line=0.0,
                          log_scale=False,
                          favors=favors,
                          favors_label=favors_label,
                          favors_spacing=favors_spacing,
                          step=True,
                          ax=ax)

        # Returning the plot to user
        return ax

    def plot_multi_span(self, ax=None, color='k', favors=True, favors_label=("A=2", "A=0"), favors_spacing="\t"):
        # Estimating risks and their CI's
        if self.multi_span is None:
            self.estimate_multi_span()
        r = self.multi_span.reset_index()

        # Drawing plot
        ax = twister_plot(data=r, point='RD-MS', lcl='RD-MS_LCL', ucl='RD-MS_UCL', time='Time',
                          color=color,
                          reference_line=0.0,
                          log_scale=False,
                          favors=favors,
                          favors_label=favors_label,
                          favors_spacing=favors_spacing,
                          step=True,
                          ax=ax)

        # Returning the plot to user
        return ax

    def _get_variable_arrays_(self):
        t = np.asarray((self.data[self.time]))
        delta = np.asarray((self.data[self.delta]))
        a = np.asarray((self.data[self.action]))
        s = np.asarray((self.data[self.sample]))
        if self.censor is None:
            c = np.asarray(1 - self.data[self.delta])
        else:
            c = np.asarray(self.data[self.censor])
        return t, delta, a, s, c

    def _get_time_matrices_(self, t, action=None, sample=None):
        if sample is None:
            unique_times = list(np.unique(self.data.loc[self.data[self.delta] == 1, self.time]))
        else:
            if action is None:
                unique_times = list(np.unique(self.data.loc[(self.data[self.delta] == 1)
                                                            & (self.data[self.sample] == sample),
                                                            self.time]))
            else:
                unique_times = list(np.unique(self.data.loc[(self.data[self.delta] == 1)
                                                            & (self.data[self.action] == action)
                                                            & (self.data[self.sample] == sample),
                                                            self.time]))
        unique_times = np.asarray(unique_times)

        # Creating design matrices
        time_design_matrix = np.identity(n=len(unique_times))
        time_design_matrix[:, 0] = 1

        # Creating other arrays
        r_matrix = (t >= unique_times[:, None]).astype(int)
        r_star_matrix = (t == unique_times[:, None]).astype(int)
        return time_design_matrix, r_star_matrix, r_matrix, unique_times

    def _get_censor_matrices_(self, t, c, sample):
        unique_censor_times = list(np.unique(self.data.loc[(c == 1) & (self.data[self.sample] == sample), self.time]))
        unique_censor_times = np.asarray(unique_censor_times)

        # Creating design matrices
        ltfu_design_matrix = np.identity(n=len(unique_censor_times))
        ltfu_design_matrix[:, 0] = 1

        # Creating other arrays
        r_matrix = (t >= unique_censor_times[:, None]).astype(int)
        r_star_matrix = (t == unique_censor_times[:, None]).astype(int)
        r_matrix = r_matrix - (1-c)*r_star_matrix

        return ltfu_design_matrix, r_star_matrix, r_matrix, unique_censor_times

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

    def _generate_inits_plr_(self, n_covs, event, s, a=None):
        # Subset the data
        if a is None:
            ds = self.data.loc[self.data[self.sample] == s].copy()
        else:
            ds = self.data.loc[(self.data[self.sample] == s) & (self.data[self.action] == a)].copy()

        # Initial setup of variables
        d = np.asarray(ds[event])
        t = np.asarray(ds[self.time])

        # Getting unique times for the input event
        unique_times = list(np.unique(ds.loc[d == 1, self.time]))
        unique_times = np.asarray(unique_times)

        # Matrix of those in risk set by time interval
        risk_set_matrix = t[:, None] >= unique_times
        n_risk_set = np.sum(risk_set_matrix, axis=0)
        event_matrix = (t[:, None] == unique_times) * d[:, None]
        n_events = np.sum(event_matrix, axis=0)

        # Calculating the probability for each interval
        pr = n_events / n_risk_set
        pr = logit(pr)
        pr = pr - np.asarray([0., ] + [pr[0], ] * (len(pr) - 1))
        if pr[0] <= -5:                   # Bounding the intercept so it's not too low
            pr[0] = -4.6                  # ... expit(-4.6) ~ 0.01

        # Replacing infs and nans
        pr = np.nan_to_num(pr, 0.)
        pr[pr == np.inf] = 0

        # Returning list of pre-set starting values
        return [0., ]*n_covs + list(pr)

    def _generate_inits_risk_(self, s, a):
        # Getting unique times for the input event
        unique_times = list(np.unique(self.data.loc[self.data[self.delta] == 1, self.time]))
        unique_times = np.asarray(unique_times)

        # Subset the data
        ds = self.data.loc[(self.data[self.sample] == s) & (self.data[self.action] == a)].copy()
        d = np.asarray(ds[self.delta])
        t = np.asarray(ds[self.time])

        # Matrix of those in risk set by time interval
        risk_set_matrix = t[:, None] >= unique_times
        n_risk_set = np.sum(risk_set_matrix, axis=0)
        event_matrix = (t[:, None] == unique_times) * d[:, None]
        n_events = np.sum(event_matrix, axis=0)

        risk = 1 - np.cumprod(1 - (n_events / n_risk_set))
        return list(np.nan_to_num(risk, nan=1))

    def _fit_mestimator_(self, estimating_functions, init, subset=None):
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
                                subset=subset)                      # ... never subset parameters
        return fmestr

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
