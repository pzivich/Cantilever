from multiprocessing.dummy.connection import families

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from cantilever.estimators.time_to_event import BridgeIPW, BridgeGComputation
from cantilever.estimators.time_to_event.utils import align_ipcw_with_event_times


@pytest.fixture
def data():
    d = pd.DataFrame()
    d['S'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d['W'] = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    d['A'] = [1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1]
    d['T'] = [2, 1, 2, 2, 1, 5, 1, 1, 2, 4, 5, 1, 5, 2, 1, 1, 5, 2, 3, 3, 3]
    d['D'] = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    d['C'] = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    d['intercept'] = 1
    d['pid'] = d.index + 1
    d['A1'] = np.where((d['A'] == 1) & (d['S'] == 0), 1, 0)
    d['A2'] = np.where((d['A'] == 2) & (d['S'] == 1), 1, 0)
    return d

@pytest.fixture
def data2():
    d = pd.DataFrame()
    d['S'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d['W'] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d['A'] = [2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    d['T'] = [1, 2, 5, 1, 2, 4, 1, 2, 4, 1, 2, 5, 1, 2, 5, 1, 3, 1, 2, 3, 4, 5, 1, 2, 2, 3, 4, 5]
    d['D'] = [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]
    d['C'] = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    d['intercept'] = 1
    d['pid'] = d.index + 1
    d['A1'] = np.where((d['A'] == 1) & (d['S'] == 0), 1, 0)
    d['A2'] = np.where((d['A'] == 2) & (d['S'] == 1), 1, 0)
    return d

@pytest.fixture
def data_long(data):
    max_t = np.max(data['T'])
    dl = pd.DataFrame(np.repeat(data.values, max_t, axis=0), columns=data.columns)
    dl['T_in'] = dl.groupby("pid")['T'].cumcount()
    dl['T_out'] = dl['T_in'] + 1
    dl['event'] = np.where(dl['T_out'] == dl['T'], dl['D'], 0)
    dl['event'] = np.where(dl['T_out'] > dl['T'], np.nan, dl['event'])
    dl['ltfu'] = np.where(dl['T_out'] == dl['T'], dl['C'], 0)
    dl['ltfu'] = np.where(dl['T_out'] > dl['T'], np.nan, dl['ltfu'])
    return dl.drop(columns=['D', 'C', 'T'])


@pytest.fixture
def data_long2(data2):
    max_t = np.max(data2['T'])
    dl = pd.DataFrame(np.repeat(data2.values, max_t, axis=0), columns=data2.columns)
    dl['T_in'] = dl.groupby("pid")['T'].cumcount()
    dl['T_out'] = dl['T_in'] + 1
    dl['event'] = np.where(dl['T_out'] == dl['T'], dl['D'], 0)
    dl['event'] = np.where(dl['T_out'] > dl['T'], np.nan, dl['event'])
    dl['ltfu'] = np.where(dl['T_out'] == dl['T'], dl['C'], 0)
    dl['ltfu'] = np.where(dl['T_out'] > dl['T'], np.nan, dl['ltfu'])
    return dl.drop(columns=['D', 'C', 'T'])


class TestTTEUtilities:

    @pytest.fixture
    def weight_process(self):
        d_times = [1, 2, 5, 7, 9]
        c_times = [2, 3, 5, 6]
        weights = [[1, 1, 1],
                   [7, 5, 9],
                   [3, 3, 8],
                   [9, 9, 10]]
        return d_times, c_times, weights

    def test_align_ipcw_with_event_times(self, weight_process):
        dt, ct, cw = weight_process

        actual = align_ipcw_with_event_times(event_time_vector=dt, censor_time_vector=ct, censor_weights=cw)
        expect = np.asarray([[1, 1, 1],
                             [1, 1, 1],
                             [7, 5, 9],
                             [9, 9, 10],
                             [9, 9, 10],
                             ]).T

        npt.assert_allclose(actual, expect, atol=1e-8)


class TestBridgeIPW:

    def test_sample_model(self, data):
        # Function in Cantilever
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.sample_model(model="W")

        # External reference
        fm = smf.glm("S ~ W", data=data, family=sm.families.Binomial()).fit()

        # Comparing coefficients
        npt.assert_allclose(bipw._sample_coefs_,
                            fm.params,
                            atol=1e-8)

    def test_action_model(self, data):
        # Function in Cantilever
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.action_model(model="W")

        # External reference
        fm1 = smf.glm("A2 ~ W", data=data.loc[data['S'] == 1], family=sm.families.Binomial()).fit()
        fm0 = smf.glm("A1 ~ W", data=data.loc[data['S'] == 0], family=sm.families.Binomial()).fit()

        # Comparing coefficients
        npt.assert_allclose(bipw._action_coefs_,
                            list(fm1.params) + list(fm0.params),
                            atol=1e-8)

    def test_censor_model(self, data, data_long):
        # Function in Cantilever
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.censor_model(model="W")

        # External reference
        dl = data_long
        fm1 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 1) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()
        params = list(fm1.params)
        coefs1 = [params[-1], params[0], params[1]]
        fm0 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 0) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()
        params = list(fm0.params)
        coefs0 = [params[-1], params[0], params[2]]

        # Comparing coefficients
        npt.assert_allclose(bipw._censor_coefs_[0],
                            coefs1,
                            atol=1e-8)
        npt.assert_allclose(bipw._censor_coefs_[1],
                            coefs0,
                            atol=1e-8)

    def test_estimated_risks(self, data, data_long):
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.sample_model(model="W")
        bipw.action_model(model="W")
        bipw.censor_model(model="W")
        bipw.estimate_risks()

        # External reference
        dl = data_long

        # IOSW model
        pm = smf.glm("S ~ W", data=dl.loc[dl['T_in'] == 0], family=sm.families.Binomial()).fit()

        # IPTW model
        am1 = smf.glm("A2 ~ W", data=dl.loc[(dl['S'] == 1) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()
        am0 = smf.glm("A1 ~ W", data=dl.loc[(dl['S'] == 0) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()

        # IPCW model
        cm1 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 1) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()
        cm0 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 0) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()

        # Constructing complete weights
        dl['ss'] = pm.predict(dl)
        dl['iosw'] = np.where(dl['S'] == 1, 1, dl['ss'] / (1 - dl['ss']))

        dl['ps'] = np.where(dl['S'] == 1, am1.predict(dl), am0.predict(dl))
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 2), 1/dl['ps'], np.nan)
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 1), 1/(1-dl['ps']), dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 1), 1/dl['ps'], dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 0), 1/(1-dl['ps']), dl['iptw'])

        dl['cs'] = 1 - np.where(dl['S'] == 1, cm1.predict(dl), cm0.predict(dl))
        dl['cs'] = dl.groupby(['pid'])['cs'].cumprod()
        dl['cs'] = dl.groupby('pid')['cs'].shift(1).fillna(1)
        dl['ipcw'] = 1 / dl['cs']

        dl['ipw'] = dl['iosw'] * dl['iptw'] * dl['ipcw']

        def predict_at_times(kaplan_meier):
            t_eval = np.array([0, 1, 2, 3, 4, 5])
            idx = np.searchsorted(kaplan_meier.surv_times, t_eval, side="right") - 1
            surv_at_t = np.where(idx >= 0, kaplan_meier.surv_prob[idx], 1.0)
            return 1 - surv_at_t

        # Kaplan-Meier estimator
        dl21 = dl.loc[(dl['A'] == 2) & (dl['S'] == 1)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl21["T_out"], dl21["event"], entry=dl21['T_in'], freq_weights=dl21['ipw'])
        risk21 =  predict_at_times(km)

        dl11 = dl.loc[(dl['A'] == 1) & (dl['S'] == 1)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl11["T_out"], dl11["event"], entry=dl11['T_in'], freq_weights=dl11['ipw'])
        risk11 =  predict_at_times(km)

        dl10 = dl.loc[(dl['A'] == 1) & (dl['S'] == 0)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl10["T_out"], dl10["event"], entry=dl10['T_in'], freq_weights=dl10['ipw'])
        risk10 =  predict_at_times(km)

        dl00 = dl.loc[(dl['A'] == 0) & (dl['S'] == 0)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl00["T_out"], dl00["event"], entry=dl00['T_in'], freq_weights=dl00['ipw'])
        risk00 =  predict_at_times(km)

        # Comparisons to by-hand with long data set
        npt.assert_allclose(bipw.risks['R-A2S1'], risk21, atol=1e-7)
        npt.assert_allclose(bipw.risks['R-A1S1'], risk11, atol=1e-7)
        npt.assert_allclose(bipw.risks['R-A1S0'], risk10, atol=1e-7)
        npt.assert_allclose(bipw.risks['R-A0S0'], risk00, atol=1e-7)

    def test_diagnostic(self, data, data_long):
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.sample_model(model="W")
        bipw.action_model(model="W")
        bipw.censor_model(model="W")
        bipw.estimate_diagnostic()

        # External reference
        dl = data_long

        # IOSW model
        pm = smf.glm("S ~ W", data=dl.loc[dl['T_in'] == 0], family=sm.families.Binomial()).fit()

        # IPTW model
        am1 = smf.glm("A2 ~ W", data=dl.loc[(dl['S'] == 1) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()
        am0 = smf.glm("A1 ~ W", data=dl.loc[(dl['S'] == 0) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()

        # IPCW model
        cm1 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 1) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()
        cm0 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 0) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()

        # Constructing complete weights
        dl['ss'] = pm.predict(dl)
        dl['iosw'] = np.where(dl['S'] == 1, 1, dl['ss'] / (1 - dl['ss']))

        dl['ps'] = np.where(dl['S'] == 1, am1.predict(dl), am0.predict(dl))
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 2), 1/dl['ps'], np.nan)
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 1), 1/(1-dl['ps']), dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 1), 1/dl['ps'], dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 0), 1/(1-dl['ps']), dl['iptw'])

        dl['cs'] = 1 - np.where(dl['S'] == 1, cm1.predict(dl), cm0.predict(dl))
        dl['cs'] = dl.groupby(['pid'])['cs'].cumprod()
        dl['cs'] = dl.groupby('pid')['cs'].shift(1).fillna(1)
        dl['ipcw'] = 1 / dl['cs']

        dl['ipw'] = dl['iosw'] * dl['iptw'] * dl['ipcw']

        def predict_at_times(kaplan_meier):
            t_eval = np.array([0, 1, 2, 3, 4, 5])
            idx = np.searchsorted(kaplan_meier.surv_times, t_eval, side="right") - 1
            surv_at_t = np.where(idx >= 0, kaplan_meier.surv_prob[idx], 1.0)
            return 1 - surv_at_t

        # Kaplan-Meier estimator
        dl11 = dl.loc[(dl['A'] == 1) & (dl['S'] == 1)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl11["T_out"], dl11["event"], entry=dl11['T_in'], freq_weights=dl11['ipw'])
        risk11 =  predict_at_times(km)

        dl10 = dl.loc[(dl['A'] == 1) & (dl['S'] == 0)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl10["T_out"], dl10["event"], entry=dl10['T_in'], freq_weights=dl10['ipw'])
        risk10 =  predict_at_times(km)

        # Comparisons to by-hand with long data set
        npt.assert_allclose(bipw.diagnostic['RD-D'], risk11 - risk10, atol=1e-7)

    def test_diagnostic_ird(self, data, data_long):
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.sample_model(model="W")
        bipw.action_model(model="W")
        bipw.censor_model(model="W")
        bipw.estimate_diagnostic()

        # External reference
        dl = data_long

        # IOSW model
        pm = smf.glm("S ~ W", data=dl.loc[dl['T_in'] == 0], family=sm.families.Binomial()).fit()

        # IPTW model
        am1 = smf.glm("A2 ~ W", data=dl.loc[(dl['S'] == 1) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()
        am0 = smf.glm("A1 ~ W", data=dl.loc[(dl['S'] == 0) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()

        # IPCW model
        cm1 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 1) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()
        cm0 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 0) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()

        # Constructing complete weights
        dl['ss'] = pm.predict(dl)
        dl['iosw'] = np.where(dl['S'] == 1, 1, dl['ss'] / (1 - dl['ss']))

        dl['ps'] = np.where(dl['S'] == 1, am1.predict(dl), am0.predict(dl))
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 2), 1/dl['ps'], np.nan)
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 1), 1/(1-dl['ps']), dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 1), 1/dl['ps'], dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 0), 1/(1-dl['ps']), dl['iptw'])

        dl['cs'] = 1 - np.where(dl['S'] == 1, cm1.predict(dl), cm0.predict(dl))
        dl['cs'] = dl.groupby(['pid'])['cs'].cumprod()
        dl['cs'] = dl.groupby('pid')['cs'].shift(1).fillna(1)
        dl['ipcw'] = 1 / dl['cs']

        dl['ipw'] = dl['iosw'] * dl['iptw'] * dl['ipcw']

        def predict_at_times(kaplan_meier):
            t_eval = np.array([0, 1, 2, 3, 4, 5])
            idx = np.searchsorted(kaplan_meier.surv_times, t_eval, side="right") - 1
            surv_at_t = np.where(idx >= 0, kaplan_meier.surv_prob[idx], 1.0)
            return 1 - surv_at_t

        # Kaplan-Meier estimator
        dl11 = dl.loc[(dl['A'] == 1) & (dl['S'] == 1)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl11["T_out"], dl11["event"], entry=dl11['T_in'], freq_weights=dl11['ipw'])
        risk11 =  predict_at_times(km)

        dl10 = dl.loc[(dl['A'] == 1) & (dl['S'] == 0)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl10["T_out"], dl10["event"], entry=dl10['T_in'], freq_weights=dl10['ipw'])
        risk10 =  predict_at_times(km)

        delta_rd = risk11[1:] - risk10[1:]
        delta_t = [1-0, 2-1, 3-2, 4-3, 5-4]
        ird = np.sum(delta_rd * delta_t)

        # Comparisons to by-hand with long data set
        npt.assert_allclose(bipw.diagnostic_test['IRD'], ird, atol=1e-7)

    def test_single_span(self, data, data_long):
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.sample_model(model="W")
        bipw.action_model(model="W")
        bipw.censor_model(model="W")
        bipw.estimate_risks()
        bipw.estimate_single_span()

        # External reference
        dl = data_long

        # IOSW model
        pm = smf.glm("S ~ W", data=dl.loc[dl['T_in'] == 0], family=sm.families.Binomial()).fit()

        # IPTW model
        am1 = smf.glm("A2 ~ W", data=dl.loc[(dl['S'] == 1) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()
        am0 = smf.glm("A1 ~ W", data=dl.loc[(dl['S'] == 0) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()

        # IPCW model
        cm1 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 1) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()
        cm0 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 0) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()

        # Constructing complete weights
        dl['ss'] = pm.predict(dl)
        dl['iosw'] = np.where(dl['S'] == 1, 1, dl['ss'] / (1 - dl['ss']))

        dl['ps'] = np.where(dl['S'] == 1, am1.predict(dl), am0.predict(dl))
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 2), 1/dl['ps'], np.nan)
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 1), 1/(1-dl['ps']), dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 1), 1/dl['ps'], dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 0), 1/(1-dl['ps']), dl['iptw'])

        dl['cs'] = 1 - np.where(dl['S'] == 1, cm1.predict(dl), cm0.predict(dl))
        dl['cs'] = dl.groupby(['pid'])['cs'].cumprod()
        dl['cs'] = dl.groupby('pid')['cs'].shift(1).fillna(1)
        dl['ipcw'] = 1 / dl['cs']

        dl['ipw'] = dl['iosw'] * dl['iptw'] * dl['ipcw']

        def predict_at_times(kaplan_meier):
            t_eval = np.array([0, 1, 2, 3, 4, 5])
            idx = np.searchsorted(kaplan_meier.surv_times, t_eval, side="right") - 1
            surv_at_t = np.where(idx >= 0, kaplan_meier.surv_prob[idx], 1.0)
            return 1 - surv_at_t

        # Kaplan-Meier estimator
        dl21 = dl.loc[(dl['A'] == 2) & (dl['S'] == 1)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl21["T_out"], dl21["event"], entry=dl21['T_in'], freq_weights=dl21['ipw'])
        risk21 =  predict_at_times(km)

        dl00 = dl.loc[(dl['A'] == 0) & (dl['S'] == 0)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl00["T_out"], dl00["event"], entry=dl00['T_in'], freq_weights=dl00['ipw'])
        risk00 =  predict_at_times(km)

        # Comparisons to by-hand with long data set
        npt.assert_allclose(bipw.single_span['RD-SS'], risk21 - risk00, atol=1e-7)

    def test_multi_span(self, data, data_long):
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.sample_model(model="W")
        bipw.action_model(model="W")
        bipw.censor_model(model="W")
        bipw.estimate_multi_span()

        # External reference
        dl = data_long

        # IOSW model
        pm = smf.glm("S ~ W", data=dl.loc[dl['T_in'] == 0], family=sm.families.Binomial()).fit()

        # IPTW model
        am1 = smf.glm("A2 ~ W", data=dl.loc[(dl['S'] == 1) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()
        am0 = smf.glm("A1 ~ W", data=dl.loc[(dl['S'] == 0) & (dl['T_in'] == 0)], family=sm.families.Binomial()).fit()

        # IPCW model
        cm1 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 1) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()
        cm0 = smf.glm("ltfu ~ W + C(T_in)", data=dl.loc[(dl['S'] == 0) & (dl['event'] == 0)],
                      family=sm.families.Binomial()).fit()

        # Constructing complete weights
        dl['ss'] = pm.predict(dl)
        dl['iosw'] = np.where(dl['S'] == 1, 1, dl['ss'] / (1 - dl['ss']))

        dl['ps'] = np.where(dl['S'] == 1, am1.predict(dl), am0.predict(dl))
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 2), 1/dl['ps'], np.nan)
        dl['iptw'] = np.where((dl['S'] == 1) & (dl['A'] == 1), 1/(1-dl['ps']), dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 1), 1/dl['ps'], dl['iptw'])
        dl['iptw'] = np.where((dl['S'] == 0) & (dl['A'] == 0), 1/(1-dl['ps']), dl['iptw'])

        dl['cs'] = 1 - np.where(dl['S'] == 1, cm1.predict(dl), cm0.predict(dl))
        dl['cs'] = dl.groupby(['pid'])['cs'].cumprod()
        dl['cs'] = dl.groupby('pid')['cs'].shift(1).fillna(1)
        dl['ipcw'] = 1 / dl['cs']

        dl['ipw'] = dl['iosw'] * dl['iptw'] * dl['ipcw']

        def predict_at_times(kaplan_meier):
            t_eval = np.array([0, 1, 2, 3, 4, 5])
            idx = np.searchsorted(kaplan_meier.surv_times, t_eval, side="right") - 1
            surv_at_t = np.where(idx >= 0, kaplan_meier.surv_prob[idx], 1.0)
            return 1 - surv_at_t

        # Kaplan-Meier estimator
        dl21 = dl.loc[(dl['A'] == 2) & (dl['S'] == 1)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl21["T_out"], dl21["event"], entry=dl21['T_in'], freq_weights=dl21['ipw'])
        risk21 =  predict_at_times(km)

        dl11 = dl.loc[(dl['A'] == 1) & (dl['S'] == 1)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl11["T_out"], dl11["event"], entry=dl11['T_in'], freq_weights=dl11['ipw'])
        risk11 =  predict_at_times(km)

        dl10 = dl.loc[(dl['A'] == 1) & (dl['S'] == 0)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl10["T_out"], dl10["event"], entry=dl10['T_in'], freq_weights=dl10['ipw'])
        risk10 =  predict_at_times(km)

        dl00 = dl.loc[(dl['A'] == 0) & (dl['S'] == 0)].dropna(subset='event').copy()
        km = sm.SurvfuncRight(dl00["T_out"], dl00["event"], entry=dl00['T_in'], freq_weights=dl00['ipw'])
        risk00 =  predict_at_times(km)

        # Comparisons to by-hand with long data set
        npt.assert_allclose(bipw.multi_span['RD-MS'], (risk21 - risk11) + (risk10 - risk00), atol=1e-7)


class TestBridgeGComputation:

    def test_outcome_model(self, data2, data_long2):
        bgcomp = BridgeGComputation(data=data2, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bgcomp.outcome_model(model="W")

        # External reference
        dl = data_long2
        combos = [[1, 2],  [1, 1],
                  [0, 1], [0, 0]]
        for i in [0, 1, 2, 3]:
            s, a = combos[i]
            dls = dl.loc[(dl['S'] == s) & (dl['A'] == a)]
            unique_times = np.unique(dls.loc[dls['event'] == 1, 'T_in'])
            fm = smf.glm("event ~ W + C(T_in)", data=dls.loc[dls['T_in'].isin(unique_times)],
                         family=sm.families.Binomial()).fit()
            params = list(fm.params)
            coefs = [params[-1], ] + list(params[:-1])

            # Checking coefficients for subset model
            npt.assert_allclose(bgcomp._outcome_coefs_[i], coefs, atol=1e-7)

    def test_estimated_risks(self, data2, data_long2):
        bgcomp = BridgeGComputation(data=data2, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bgcomp.outcome_model(model="W")
        bgcomp.estimate_risks()

        # External reference
        dl = data_long2
        dl1 = dl.loc[dl['S'] == 1].copy()
        combos = [[1, 2],  [1, 1],
                  [0, 1], [0, 0]]
        risks = []
        for i in [0, 1, 2, 3]:
            s, a = combos[i]
            dls = dl.loc[(dl['S'] == s) & (dl['A'] == a)]
            fm = smf.glm("event ~ W + C(T_in)", data=dls, family=sm.families.Binomial()).fit()

            dl1['csurv_i'] = 1 - fm.predict(dl1)
            dl1['surv_i'] = dl1.groupby(['pid'])['csurv_i'].cumprod()
            risk = 1 - dl1.groupby('T_out')['surv_i'].mean()
            risks.append(risk)

        # Checking coefficients for subset model
        npt.assert_allclose(bgcomp.risks['R-A2S1'], [0., ] + list(risks[0]), atol=1e-7)
        npt.assert_allclose(bgcomp.risks['R-A1S1'], [0., ] + list(risks[1]), atol=1e-7)
        npt.assert_allclose(bgcomp.risks['R-A1S0'], [0., ] + list(risks[2]), atol=1e-7)
        npt.assert_allclose(bgcomp.risks['R-A0S0'], [0., ] + list(risks[3]), atol=1e-7)

    def test_diagnostic(self, data2, data_long2):
        bgcomp = BridgeGComputation(data=data2, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bgcomp.outcome_model(model="W")
        bgcomp.estimate_diagnostic()

        # External reference
        dl = data_long2
        dl1 = dl.loc[dl['S'] == 1].copy()
        combos = [[1, 1], [0, 1]]
        risks = []
        for i in [0, 1]:
            s, a = combos[i]
            dls = dl.loc[(dl['S'] == s) & (dl['A'] == a)]
            fm = smf.glm("event ~ W + C(T_in)", data=dls, family=sm.families.Binomial()).fit()

            dl1['csurv_i'] = 1 - fm.predict(dl1)
            dl1['surv_i'] = dl1.groupby(['pid'])['csurv_i'].cumprod()
            risk = 1 - dl1.groupby('T_out')['surv_i'].mean()
            risks.append(risk)

        # Checking coefficients for subset model
        npt.assert_allclose(bgcomp.diagnostic['RD-D'], [0., ] + list(risks[0] - risks[1]), atol=1e-7)

    def test_diagnostic_ird(self, data2, data_long2):
        bgcomp = BridgeGComputation(data=data2, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bgcomp.outcome_model(model="W")
        bgcomp.estimate_diagnostic()

        # External reference
        dl = data_long2
        dl1 = dl.loc[dl['S'] == 1].copy()
        combos = [[1, 1], [0, 1]]
        risks = []
        for i in [0, 1]:
            s, a = combos[i]
            dls = dl.loc[(dl['S'] == s) & (dl['A'] == a)]
            fm = smf.glm("event ~ W + C(T_in)", data=dls, family=sm.families.Binomial()).fit()

            dl1['csurv_i'] = 1 - fm.predict(dl1)
            dl1['surv_i'] = dl1.groupby(['pid'])['csurv_i'].cumprod()
            risk = 1 - dl1.groupby('T_out')['surv_i'].mean()
            risks.append(risk)

        delta_rd = risks[0] - risks[1]
        delta_t = [1-0, 2-1, 3-2, 4-3, 5-4]
        ird = np.sum(delta_rd * delta_t)

        # Comparisons to by-hand with long data set
        npt.assert_allclose(bgcomp.diagnostic_test['IRD'], ird, atol=1e-7)

    def test_single_span(self, data2, data_long2):
        bgcomp = BridgeGComputation(data=data2, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bgcomp.outcome_model(model="W")
        bgcomp.estimate_single_span()

        # External reference
        dl = data_long2
        dl1 = dl.loc[dl['S'] == 1].copy()
        combos = [[1, 2], [0, 0]]
        risks = []
        for i in [0, 1]:
            s, a = combos[i]
            dls = dl.loc[(dl['S'] == s) & (dl['A'] == a)]
            fm = smf.glm("event ~ W + C(T_in)", data=dls, family=sm.families.Binomial()).fit()

            dl1['csurv_i'] = 1 - fm.predict(dl1)
            dl1['surv_i'] = dl1.groupby(['pid'])['csurv_i'].cumprod()
            risk = 1 - dl1.groupby('T_out')['surv_i'].mean()
            risks.append(risk)

        # Checking coefficients for subset model
        npt.assert_allclose(bgcomp.single_span['RD-SS'], [0., ] + list(risks[0] - risks[1]), atol=1e-7)

    def test_multi_span(self, data2, data_long2):
        bgcomp = BridgeGComputation(data=data2, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bgcomp.outcome_model(model="W")
        bgcomp.estimate_multi_span()

        # External reference
        dl = data_long2
        dl1 = dl.loc[dl['S'] == 1].copy()
        combos = [[1, 2], [1, 1], [0, 1], [0, 0]]
        risks = []
        for i in [0, 1, 2, 3]:
            s, a = combos[i]
            dls = dl.loc[(dl['S'] == s) & (dl['A'] == a)]
            fm = smf.glm("event ~ W + C(T_in)", data=dls, family=sm.families.Binomial()).fit()

            dl1['csurv_i'] = 1 - fm.predict(dl1)
            dl1['surv_i'] = dl1.groupby(['pid'])['csurv_i'].cumprod()
            risk = 1 - dl1.groupby('T_out')['surv_i'].mean()
            risks.append(risk)

        # Checking coefficients for subset model
        npt.assert_allclose(bgcomp.multi_span['RD-MS'],
                            [0., ] + list((risks[0] - risks[1]) + (risks[2] - risks[3])),
                            atol=1e-7)


class TestBridgeAIPW:

    def test_sample_model(self, data):
        # Function in Cantilever
        bipw = BridgeIPW(data=data, time='T', delta='D', action='A', sample='S', censor='C', verbose=False)
        bipw.sample_model(model="W")

        # External reference
        fm = smf.glm("S ~ W", data=data, family=sm.families.Binomial()).fit()

        # Comparing coefficients
        npt.assert_allclose(bipw._sample_coefs_,
                            fm.params,
                            atol=1e-8)
