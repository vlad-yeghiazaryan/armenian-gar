#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[104]:


# main
import pandas as pd
import numpy as np
import operator
import re
from scipy import stats

# models
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LinearRegression

# unsupervised models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# diagnositcs and metrics
import shap
from statsmodels.iolib.summary2 import summary_col
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

# preprocessing
from feature_engine.timeseries.forecasting import LagFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.tsa.tsatools import add_trend

# feature selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

# tests
from statsmodels.tsa.stattools import adfuller, coint

# charts
import matplotlib.pyplot as plt
import matplotlib.colors as matcolors
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

# models (import)
from GAR.growth import calc_growth_rate
from GAR.quantfit import run_quantfit
from GAR.tsfit import dist_fit, gen_PDF_and_CDF
from GAR.historical import select_x_list

# util
from tqdm.notebook import tqdm
from dateutil import parser
from IPython.display import display


# In[3]:


class StandardScalerDF:

    def __init__(self, mean=None, std=None):
        self.mean_ = mean
        self.std_ = std

    def fit(self, X):
        self.mean_ = X.mean()
        self.std_ = X.std()
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fit yet.")
        scaled_X = (X - self.mean_) / self.std_
        return scaled_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fit yet.")
        X = X_scaled * self.std_ + self.mean_
        return X


# In[4]:


# Kalman filter: Kalman_companion method below
def kalman_companion(data, S0, P0, L, R, T, Q):
    # scalar setup
    t, nvars = data.shape
    nstates_all = S0.shape[0]
    nstates = L.shape[0]

    # matrix setup
    S_pred = S0
    P_pred = P0
    S = np.zeros((data.shape[0], nstates_all))
    P = np.zeros((nstates_all**2, t))

    # Kalman filter step, extracting states
    for index, (date, y_i) in enumerate(data.iterrows()):
        nu = np.array(y_i) - L.T @ S_pred[:nstates]  # conditional forecast error
        f = L.T @ P_pred[:nstates, :nstates] @ L + R  # variance of the conditional forecast error
        finv = L @ np.linalg.inv(f)
        Stt = S_pred + (P_pred[:, :nstates] @ finv @ nu)
        Ptt = P_pred - P_pred[:, :nstates] @ finv @ (L.T @ P_pred[:nstates, :])
        if date < data.index[-1]:
            S_pred = T @ Stt
            P_pred = T @ Ptt @ T.T + Q
        S[index, :] = Stt
        P[:, index] = Ptt.reshape(-1)

    # Now we can draw Sdraw(T|T) ~ N(S(T|T), P(T|T))
    S_draw = np.zeros((t, nstates_all))
    S_draw[t-1, :] = S[t-1, :]

    # iterate 'down', drawing at each step, use modification for singular Q
    T_star = T[:nstates, :]
    Q_star = Q[:nstates, :nstates]

    # Kalman Smoothing step
    for i in range(1, t)[::-1]:
        Sf = S_draw[i, :nstates]
        Stt = S[i-1, :]
        Ptt = P[:, i-1].reshape(nstates_all, -1)
        f = T_star @ Ptt @ T_star.T + Q_star
        finv = T_star.T @ np.linalg.inv(f)
        nu = Sf - T_star @ Stt
        Smean = Stt + Ptt @ finv @ nu
        # Svar = Ptt - Ptt @ finv @ (T_star @ Ptt)
        S_draw[i-1, :] = Smean
    S_draw = S_draw[:, :nstates]
    return S_draw


# In[5]:


def Minn_prior_KOOP(gamma, nstates, nlags, nvars):
    # This is the version of the Minnesota prior with no dependence on the
    # standard deviations of the univariate regressions. This prior allows
    # online estimation and forecasting of the large TVP-VAR.

    # 1. Minnesota Mean on VAR regression coefficients
    A_prior = np.concatenate((0.9*np.eye(nstates),
                              np.zeros(((nlags-1)*nstates, nstates))), axis=0).T
    a_prior = A_prior.T.reshape(-1)

    # 2. Minnesota Variance on VAR regression coefficients
    # Create an array of dimensions nvars x nstates, which will contain the nvars diagonal
    # elements of the covariance matrix, in each of the nstates equations.
    V_i = np.zeros((int(nvars/nstates), nstates))

    for i in range(nstates):
        for j in range(int(nvars/nstates)):
            V_i[j, i] = gamma / (np.ceil((j+1)/nstates))**2

    # 3. Now V (MINNESOTA VARIANCE) is a diagonal matrix with diagonal elements of V_i
    V_prior = np.diag(V_i.reshape(-1))  # this is the prior variance of the vector alpha
    return a_prior, V_prior


# In[6]:


def create_RHS_NI(YY, nstates, nlags, t):
    K = nlags*(nstates**2)  # K is the number of elements in the state vector
    # Create x_t matrix
    # first find the zeros in matrix x_t
    x_t = np.zeros(((t-nlags)*nstates, K))
    for i in range(t-nlags):
        ztemp = np.array([]).reshape(nstates, -1)
        for j in range(nlags):
            xtemp = YY[i, j*nstates:(j+1)*nstates]
            xtemp = np.kron(np.eye(nstates), xtemp)
            ztemp = np.concatenate((ztemp, xtemp), axis=1)
        x_t[i*nstates:(i+1)*nstates, :] = ztemp
    return x_t, K


# In[7]:


class TVP_FAVAR():
    # Function to estimate time-varying loadings, coefficients, and covariances
    # from a TVP-FAVAR, conditional on feeding in an estimate of the factors
    # (Principal Components). This function runs the Kalman filter and smoother
    # for all time-varying parameters using an adaptive algorithm (EWMA filter
    # for the covariances).
    # Written by Dimitris Korobilis, 2012
    def __init__(self, endog, exog, nlags):
        nstates = exog.shape[1]
        t, nvars = endog.shape
        self.nlags = nlags
        self.nmacro = nstates - nfactors
        nstates_all = nstates*self.nlags
        nVAR = self.nlags*(nstates**2)

        # setup: lambda
        self.lambda_pred = np.zeros((nvars, nstates, t))
        self.lambda_update = np.zeros((nvars, nstates, t))

        for i in range(t):
            self.lambda_pred[:nstates, :nstates, i] = np.eye(nstates)
            self.lambda_update[:nstates, :nstates, i] = np.eye(nstates)

        # setup: beta
        self.beta_pred = np.zeros((nVAR, t))
        self.beta_update = np.zeros((nVAR, t))

        # setup: states
        self.Rl_t = np.zeros((nstates, nstates, nvars, t))
        self.Sl_t = np.zeros((nstates, nstates, nvars, t))
        self.Rb_t = np.zeros((nVAR, nVAR, t))
        self.Sb_t = np.zeros((nVAR, nVAR, t))

        # setup: expectations
        self.x_t_pred = np.zeros((t, nvars))
        self.e_t = np.zeros((nvars, t))
        self.lambda_t = np.zeros((nvars, nstates, t))
        self.beta_t = np.zeros((nstates_all, nstates_all, t))

        # setup: covariance
        self.Q_t = np.zeros((nstates, nstates, t))
        self.V_t = np.zeros((nvars, nvars, t))

        # define lags of the factors to be used in the state (VAR) equation
        lf = LagFeatures(periods=list(range(1, self.nlags+1)), drop_original=True)
        xx = lf.fit_transform(exog).dropna()
        yy = exog.loc[xx.index].copy()
        self.xx = np.array(xx)
        self.yy = np.array(yy)
        self.exog = np.array(exog)
        self.endog = np.array(endog)
        Flagtemp, m = create_RHS_NI(self.xx.copy(), nstates, self.nlags, t)
        self.Flag = np.concatenate((np.zeros((nstates_all, nVAR)), Flagtemp), axis=0)

    def kalman_setup(self, priors):
        # init
        t, nvars = self.endog.shape
        self.Q_t[:, :, 0] = priors['Q_0']
        self.V_t[:, :, 0] = np.diag(np.diag(priors['V_0']))
        self.lambda_pred[:, :, 0] = priors['lambda_0_mean']
        self.beta_pred[:, 0] = priors['beta_0_mean']
        self.beta_update[:, 0] = self.beta_pred[:, 0]
        self.Rb_t[:, :, 0] = priors['beta_0_var']
        for i in range(self.nmacro, nvars):
            self.Rl_t[:, :, i, 0] = priors['lambda_0_var']

    def kalman_update_coeffs(self, t):
        nstates = self.exog.shape[1]
        # Update VAR coefficients conditional on Principal Componets estimates
        Rx = self.Rb_t[:, :, t] @ self.Flag[t*nstates:(t+1)*nstates, :].T
        KV_b = self.Q_t[:, :, t] + self.Flag[t*nstates:(t+1)*nstates, :]  @ Rx
        KG = Rx @ np.linalg.inv(KV_b)
        beta_adj = KG @ (self.exog[t, :].T - self.Flag[t*nstates:(t+1)*nstates, :] @ self.beta_pred[:, t])
        self.beta_update[:, t] = self.beta_pred[:, t] + beta_adj
        Sb_adj = KG @ (self.Flag[t*nstates:(t+1)*nstates, :] @ self.Rb_t[:, :, t])
        self.Sb_t[:, :, t] = self.Rb_t[:, :, t] - Sb_adj

    def kalman_filter_early(self, t, decays, priors):
        nstates = self.exog.shape[1]
        exog_i = self.exog[t, :]

        # priors
        self.beta_pred[:, t] = priors['beta_0_mean']
        self.beta_update[:, t] = self.beta_pred[:, t]
        self.Rb_t[:, :, t] = priors['beta_0_var']

        # expectations
        Gf_t = 0.1*(exog_i.reshape(-1, 1) @ exog_i.reshape(1, -1))
        if t > 0:
            self.Q_t[:, :, t] = decays[1]*self.Q_t[:, :, t-1] + (1-decays[1])*Gf_t[:nstates, :nstates]

    def kalman_filter(self, t, decays):
        nstates = self.exog.shape[1]
        nstates_all = nstates*self.nlags
        yy_i = self.yy[t-self.nlags]
        xx_i = self.xx[t-self.nlags]

        # expectations
        ee_i = (yy_i - xx_i @ self.B[:nstates, :nstates_all].T)
        Gf_t = ee_i.reshape(-1, 1) @ ee_i.reshape(1, -1)
        self.Q_t[:, :, t] = decays[1]*self.Q_t[:, :, t-1] + (1-decays[1])*Gf_t[:nstates, :nstates]

        # update beta
        self.beta_pred[:, t] = self.beta_update[:, t-1]
        self.Rb_t[:, :, t] = (1 / decays[3]) * self.Sb_t[:, :, t-1]

    def kalman_predict(self, t, decays):
        nstates = self.exog.shape[1]
        nvars = self.endog.shape[1]
        n_undefined_rows = nstates*(self.nlags-1)
        p = self.nmacro

        # Kalman predict steps
        if t > 0:
            self.lambda_pred[:, :, t] = self.lambda_update[:, :, t-1]
            self.Rl_t[:, :, :, t] = (1/decays[2])*self.Sl_t[:, :, :, t-1]

        # One step ahead prediction based on PC factor
        self.x_t_pred[t, :] = self.lambda_pred[:, :, t] @ self.exog[t, :].T

        # Prediction error
        self.e_t[:, t] = self.endog[t, :] - self.x_t_pred[t, :]

        # Get the measurement error variance
        self.A_t = self.e_t[self.nmacro:, t].reshape(-1, 1) @ self.e_t[self.nmacro:, t].reshape(1, -1)
        if t > 0:
            self.V_t[p:, p:, t] = decays[0]*self.V_t[p:, p:, t-1] + (1-decays[0])*np.diag(np.diag(self.A_t))

        # Update loadings conditional on Principal Components estimates
        for i in range(self.nmacro, nvars):
            Rx = self.Rl_t[:nstates, :nstates, i, t] @ self.exog[t, :nstates].T
            KV_l = self.V_t[i, i, t] + self.exog[t, :nstates] @ Rx
            KG = Rx / KV_l
            lambda_adj = KG * (self.endog[t, i].T - self.lambda_pred[i, :nstates, t] @ self.exog[t, :nstates].T)
            self.lambda_update[i, :nstates, t] = self.lambda_pred[i, :nstates, t] + lambda_adj
            Sl_adj = KG.reshape(-1, 1) @ (self.exog[t, :nstates] @ self.Rl_t[:nstates, :nstates, i, t]).reshape(1, -1)
            self.Sl_t[:nstates, :nstates, i, t] = self.Rl_t[:nstates, :nstates, i, t] - Sl_adj

        # update coeffs
        if t >= self.nlags:
            self.kalman_update_coeffs(t)

        # Assign coefficients
        biga = self.beta_update[:, t].reshape(nstates, self.nlags, nstates).transpose(1, 0, 2).reshape(nstates, -1)
        b_other = np.concatenate((np.eye(n_undefined_rows), np.zeros((n_undefined_rows, nstates))), axis=1)
        self.B = np.concatenate((biga, b_other), axis=0)
        self.lambda_t[:, :, t] = self.lambda_update[:, :, t]
        if np.max(np.abs(np.linalg.eig(self.B)[0])) < 0.9999:
            self.beta_t[:, :, t] = self.B
        else:
            self.beta_t[:, :, t] = self.beta_t[:, :, t-1]
            self.beta_update[:, t] = 0.95*self.beta_update[:, t-1]

    def smoother_setup(self, t):
        self.lambda_new = 0*self.lambda_update
        self.lambda_new[:, :, t] = self.lambda_update[:, :, t]
        self.beta_new = 0*self.beta_update
        self.beta_new[:, t] = self.beta_update[:, t]
        self.Q_t_new = 0*self.Q_t
        self.Q_t_new[:, :, t] = self.Q_t[:, :, t]
        self.V_t_new = 0*self.V_t
        self.V_t_new[:, :, t] = self.V_t[:, :, t]

    def kalman_smoother(self, t):
        # init
        nstates = self.exog.shape[1]
        nvars = self.endog.shape[1]
        p = self.nmacro

        # smooth lambda
        self.lambda_new[:nstates, :, t] = self.lambda_update[:nstates, :, t]
        for i in range(nstates, nvars):
            Ul_t = self.Sl_t[:nstates, :nstates, i, t] @ np.linalg.inv(self.Rl_t[:nstates, :nstates, i, t+1])
            lambda_adj = (self.lambda_new[i, :nstates, t+1] - self.lambda_pred[i, :nstates, t+1]) @ Ul_t.T
            self.lambda_new[i, :nstates, t] = self.lambda_update[i, :nstates, t] + lambda_adj

        # smooth beta
        if np.sum(self.Rb_t[:, :, t+1]) == 0:
            self.beta_new[:, t] = self.beta_update[:, t]
        else:
            Ub_t = self.Sb_t[:, :, t] @ np.linalg.inv(self.Rb_t[:, :, t+1])
            self.beta_new[:, t] = self.beta_update[:, t] + Ub_t @ (self.beta_new[:, t+1] - self.beta_pred[:, t+1])

        # smooth Q_t
        self.Q_t_new[:, :, t] = 0.9*self.Q_t[:, :, t] + 0.1*self.Q_t_new[:, :, t+1]

        # smooth V_t
        self.V_t_new[p:, p:, t] = 0.9*self.V_t[p:, p:, t] + 0.1*self.V_t_new[p:, p:, t+1]

    def kalman_reassign_coefficients(self, t):
        nstates = self.exog.shape[1]
        n_undefined_rows = nstates*(self.nlags-1)
        biga = self.beta_new[:, t].reshape(nstates, self.nlags, nstates).transpose(1, 0, 2).reshape(nstates, -1)
        b_other = np.concatenate((np.eye(n_undefined_rows), np.zeros((n_undefined_rows, nstates))), axis=1)
        B = np.concatenate((biga, b_other), axis=0)
        self.lambda_t[:, :, t] = self.lambda_new[:, :, t]
        self.beta_t[:, :, t] = B

    def fit(self, nfactors, decays, priors):
        # initialize matrices for filtering
        t = self.endog.shape[0]
        self.kalman_setup(priors)

        # perform predictions at time t=0
        self.kalman_predict(0, decays)

        # 1. Kalman filter
        for irep in range(1, t):
            # Update the state covariances,
            # get the variance of the factor and update Q[t]
            if irep <= self.nlags:
                self.kalman_filter_early(irep, decays, priors)
            else:
                self.kalman_filter(irep, decays)
            self.kalman_predict(irep, decays)

        # 2. Kalman smoother
        self.smoother_setup(t-1)
        for irep in range(t-2, -1, -1):
            self.kalman_smoother(irep)

        # using smoothed values for beta and lambda
        for irep in range(t):
            self.kalman_reassign_coefficients(irep)
        return self.beta_t, self.beta_new, self.lambda_t, self.V_t, self.Q_t

    def kalman_factor_update(self, t):
        # setup
        nstates = self.exog.shape[1]

        # One step ahead prediction based on Kalman factor
        self.x_t_predf[t, :] = self.lambda_t[:, :, t] @ self.factor_pred[:nstates, t]

        # Prediction error
        self.ef_t[:, t] = self.endog[t, :] - self.x_t_predf[t, :]

        # Update the factors conditional on the estimate of lambda_t and beta_t
        KV_f = self.V_t[:, :, t] + self.lambda_t[:, :, t] @ self.Rf_t[:nstates, :nstates, t] @ self.lambda_t[:, :, t].T
        KG = (self.Rf_t[:nstates, :nstates, t] @ self.lambda_t[:, :, t].T) @ np.linalg.inv(KV_f)
        self.factor_update[:nstates, t] = self.factor_pred[:nstates, t] + KG @ self.ef_t[:, t]
        self.Sf_t[:nstates, :nstates, t] = self.Rf_t[:nstates, :nstates, t] - KG @ (self.lambda_t[:, :, t] @ self.Rf_t[:nstates, :nstates, t])

    def factor_filter(self, priors):
        # setup
        t, nvars = self.endog.shape
        nstates = self.exog.shape[1]
        nstates_all = nstates*self.nlags

        # init
        self.factor_pred = np.zeros((nstates_all, t))
        self.factor_update = np.zeros((nstates_all, t))
        self.Rf_t = np.zeros((nstates_all, nstates_all, t))
        self.Sf_t = np.zeros((nstates_all, nstates_all, t))
        self.x_t_predf = np.zeros((t, nvars))
        self.ef_t = np.zeros((nvars, t))

        # priors
        self.factor_pred[:, 0] = priors['factor_0_mean']
        self.Rf_t[:, :, 0] = priors['factor_0_var']
        self.kalman_factor_update(0)

        # filtering step
        for irep in range(1, t):
            self.factor_pred[:, irep] = self.beta_t[:, :, irep-1] @ self.factor_update[:, irep-1]
            Sfb_t = self.beta_t[:, :, irep-1] @ self.Sf_t[:, :, irep-1] @ self.beta_t[:, :, irep-1].T
            Q_t_sparse = np.concatenate((self.Q_t[:, :, irep], np.zeros((nstates, nstates*(self.nlags-1)))), axis=1)
            Q_t_sparse = np.concatenate((Q_t_sparse, np.zeros((nstates*(self.nlags-1), nstates*nlags))), axis=0)
            self.Rf_t[:, :, irep] = Sfb_t + Q_t_sparse
            self.kalman_factor_update(irep)

    def factor_smoother(self):
        # setup
        t, nvars = self.endog.shape
        nstates = self.exog.shape[1]

        # fixed-interval smoother for the factors
        self.factor_new = 0*self.factor_update
        self.factor_new[:, t-1] = self.factor_update[:, t-1]
        self.Sf_t_new = 0*self.Sf_t
        self.Sf_t_new[:, :, t-1] = self.Sf_t[:, :, t-1]

        # smoothing
        for irep in range(t-2, -1, -1):
            self.Z_t = self.Sf_t[:, :, irep] @ self.beta_t[:, :, irep].T
            U_t = np.squeeze(self.Z_t[:nstates, :nstates] @ np.linalg.inv(self.Rf_t[:nstates, :nstates, irep+1]))
            factor_adj = U_t @ (self.factor_new[:nstates, irep+1] - self.factor_pred[:nstates, irep+1])
            self.factor_new[:nstates, irep] = self.factor_update[:nstates, irep] + factor_adj
            Sf_adj = U_t @ (self.Sf_t[:nstates, :nstates, irep+1] - self.Rf_t[:nstates, :nstates, irep+1]) @ U_t.T
            self.Sf_t_new[:nstates, :nstates, irep] = self.Sf_t[:nstates, :nstates, irep] + Sf_adj
        self.factor_new = self.factor_new[:nstates].T

    def gen_factors(self, priors):
        self.factor_filter(priors)
        self.factor_smoother()
        return self.factor_new, self.Sf_t_new


# In[8]:


def get_silhouette_scores(corr, n_init='auto'):
    ss = {k: silhouette_score(corr, KMeans(n_clusters=k, n_init=n_init).fit_predict(corr)) for k in range(2, corr.shape[0])}
    return ss


# In[9]:


# assign colours to edges depending on positive or negative correlation
def assign_colour(correlation):
    if correlation <= 0:
        return "#ffa09b"  # red
    else:
        return "#9eccb7"  # green


# assign edge thickness depending on magnitude of correlation
def assign_thickness(correlation, benchmark_thickness=2, scaling_factor=3):
    return benchmark_thickness * abs(correlation)**scaling_factor


def assign_node_size(degree, scaling_factor=50):
    return degree * scaling_factor


# define a function that "rounds" a number to 0 if it is lower than threshold, and to 1 if it is higher (or equal).
def truncate(f, threshold=0.92): 
    if (abs(f) < threshold):
        return 0
    else:
        return 1.0


# In[10]:


# 'winner takes all' method - set minium correlation threshold to remove some edges from the diagram
def plot_minimum_spanning_tree(corr, threshold=0.4, node_size=200, width=1.2,
                               node_color="#e1575c", color_legend=None,
                               title=None, title_text_size=18,
                               node_colors=None):
    font_dict = {'fontsize': title_text_size}

    # convert matrix to list of edges and rename the columns
    edges = corr.stack().reset_index()
    edges.columns = ['asset_1', 'asset_2', 'correlation']

    # remove self correlations
    edges = edges.loc[edges['asset_1'] != edges['asset_2']].copy()

    # create a new graph from edge list
    Gx = nx.from_pandas_edgelist(edges, 'asset_1', 'asset_2', edge_attr=['correlation'])

    # list to store edges to remove
    remove = []

    # loop through edges in Gx and find correlations which are below the threshold
    for asset_1, asset_2 in Gx.edges():
        corr_i = Gx[asset_1][asset_2]['correlation']
        if abs(corr_i) < threshold:
            remove.append((asset_1, asset_2))

    # remove edges contained in the remove list
    Gx.remove_edges_from(remove)

    edge_colours = []
    edge_width = []
    for key, value in nx.get_edge_attributes(Gx, 'correlation').items():
        edge_colours.append(assign_colour(value))
        edge_width.append(assign_thickness(value))

    # assign node size depending on number of connections (degree)
    node_size = []
    for key, value in dict(Gx.degree).items():
        node_size.append(assign_node_size(value))

    # create minimum spanning tree layout from Gx (after small correlations have been removed)
    mst = nx.minimum_spanning_tree(Gx)

    edge_colours = []

    # assign edge colours
    for key, value in nx.get_edge_attributes(mst, 'correlation').items():
        edge_colours.append(assign_colour(value))

    #  Using a figure to setup the legend
    has_legend = isinstance(color_legend, type(None))
    if has_legend:
        f = plt.figure(1)
        ax = f.add_subplot(1, 1, 1)
        for label, color in color_legend.items():
            ax.plot([0], [0],
                    color=color,
                    label=label)

    # draw minimum spanning tree. Set node size and width to constant
    custom_color = [node_colors[node] for node in Gx] if type(node_color) != str else node_color
    nx.draw(mst, with_labels=True, pos=nx.fruchterman_reingold_layout(mst),
            node_size=node_size, node_color=custom_color,
            edge_color=edge_colours, width=width)

    # set title
    plt.title(title, fontdict=font_dict)
    if has_legend:
        plt.legend(loc='upper right')
        f.tight_layout()


# In[11]:


def NegLog(x_i, min, min_pos):
    return np.log(((x_i - min) / (min_pos - min)) * min_pos)


def ApproxLogFunc(x_i, min, min_pos):
    if x_i >= min_pos:
        return np.log(x_i)
    else:
        return NegLog(x_i, min, min_pos)


def ApproxLog(x):
    if (x.isna().all()) or (x.sum() == 0):
        return x
    offset = x.std()
    min = x.min()
    if min > 0:
        return np.log(x)
    min = min - offset
    min_pos = np.where(x > 0, x, np.inf)
    min_pos = np.where(x >= 0, x, np.inf) if (min_pos == np.inf).all() else min_pos
    min_pos = np.NaN if (min_pos == np.inf).all() else x.iloc[min_pos.argmin()]
    min_pos = offset if min_pos == 0 else min_pos
    if (min_pos < 0) or np.isnan(min_pos):
        return -np.log(-x)
    return x.apply(lambda x_i: ApproxLogFunc(x_i, min, min_pos))


# In[12]:


def cyclicalSin(x):
    if x.min() == x.max():
        return x
    elif x.max() == 0:
        x = x + 1
    return np.sin(x * (2 * np.pi / x.max()))


def cyclicalCos(x):
    if x.min() == x.max():
        return x
    elif x.max() == 0:
        x = x + 1
    return np.cos(x * (2 * np.pi / x.max()))


def cyclicalTransformer(X):
    X_sin = X.agg(cyclicalSin).add_suffix('_sin')
    X_cos = X.agg(cyclicalCos).add_suffix('_cos')
    return pd.concat([X_sin, X_cos], axis=1)


# In[13]:


apply_op = lambda df, col, op, op_name: df.apply(lambda x: op(x, df[
    col])).add_suffix(f"_{op_name}_{col}")


def gen_relations(df, operations):
    X_rel = [
        apply_op(df, col, op, op_name)
        for col in df.columns
        for op_name, op in operations.items()
    ]
    X_rel = pd.concat(X_rel, axis=1)
    X_rel = X_rel.loc[:, ~X_rel.apply(lambda x: x.duplicated(), axis=1).all(
    )].copy()
    X_rel = pd.concat([df, X_rel], axis=1)
    return X_rel


# In[14]:


# Transformation pipeline
num_cols = selector(dtype_include=["float64", 'int64'])
generic_transformations_1 = ColumnTransformer(transformers=[
    ('Identity', FunctionTransformer(lambda X: X), num_cols),
    ('log_transform', FunctionTransformer(lambda X: X.agg(ApproxLog).add_suffix('_log')), num_cols),
    ('standardScaler', FunctionTransformer(lambda X: StandardScalerDF().fit_transform(X).add_suffix('_normalized')), num_cols),
    # ('cyclical_features', FunctionTransformer(cyclicalTransformer), num_cols)
], verbose_feature_names_out=False)
generic_transformations_1.set_output(transform='pandas')

generic_transformations_2 = ColumnTransformer(transformers=[
    ('Identity', FunctionTransformer(lambda X: X), num_cols),
    ('change_1y', FunctionTransformer(lambda X: X.diff(periods=4).add_suffix('_C_1y')), num_cols),
    ('change_2y', FunctionTransformer(lambda X: X.diff(periods=8).add_suffix('_C_2y')), num_cols),
    ('change_3y', FunctionTransformer(lambda X: X.diff(periods=12).add_suffix('_C_3y')), num_cols),
    ('pct_change_1y', FunctionTransformer(lambda X: X.pct_change(periods=4, fill_method=None).add_suffix('_PctC_1y')), num_cols),
    ('pct_change_2y', FunctionTransformer(lambda X: X.pct_change(periods=8, fill_method=None).add_suffix('_PctC_2y')), num_cols),
    ('pct_change_3y', FunctionTransformer(lambda X: X.pct_change(periods=12, fill_method=None).add_suffix('_PctC_3y')), num_cols),
], verbose_feature_names_out=False)

generic_transformations_2.set_output(transform='pandas')
operations = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': operator.truediv}
relative_transformations = FunctionTransformer(gen_relations, kw_args={'operations': operations})

# Create the final feature engineering pipeline
feature_engineering_pipeline = Pipeline([
    # Step 1: create lags
    ('lag_features', LagFeatures(periods=list(range(1, 5)), missing_values='ignore')),
    # Step 2: Perform generic transformations 1
    ('generic_transformations_1', generic_transformations_1),
    # Step 3: Perform generic transformations 2
    ('generic_transformations_2', generic_transformations_2)
    # # Step 4: Perform relative transformations
    # ('relative_transformations', relative_transformations),
])


# In[15]:


class InverseTransformer():
    def __init__(self):
        pass

    @staticmethod
    def filter_series(x, window=4):
        filtered_data = savgol_filter(x, window_length=window, polyorder=1, mode='mirror')
        return pd.Series(filtered_data, index=x.index)

    @staticmethod
    def wdist(x):
        avg = np.average(x['mean'], weights=x['weights'])
        std = np.sqrt(np.average((x['mean']-avg)**2, weights=x['weights']))
        return pd.Series({'mean': avg, 'std': std})

    def oscillation_probability(self, series_filled, actual, window=4):
        af_missing = pd.merge(series_filled, actual.isna().rename('missing'), 'left', on=['country', 'date'])
        af_missing = af_missing['missing'].fillna(True)
        afn = series_filled.groupby('country').apply(
            MinMaxScalerDF().fit_transform).reset_index(level=0, drop=True)
        afnf = afn.groupby('country').apply(self.filter_series,
                                            window=window).reset_index(level=0, drop=True)
        oscillation_score = ((afn - afnf)**2).groupby(['country', af_missing])
        counts = oscillation_score.count().groupby('missing', group_keys=False)
        weights = counts.apply(lambda x: x/np.sum(x)).rename('weights')
        oscillation_score = pd.concat([oscillation_score.mean().rename('mean'), weights], axis=1)
        osd = oscillation_score.groupby('missing').apply(inv_tf.wdist)
        osc_prob = normProbObs(osd.loc[True, 'mean'], osd.loc[False, 'mean'], osd.loc[False, 'std'])
        return 1 - osc_prob

    @staticmethod
    def inv_scale(change_rates, actual):
        ssfits = actual.groupby('country').apply(lambda x: StandardScalerDF().fit(pd.DataFrame(x)))
        ms = ssfits.apply(lambda x: pd.Series({'mean': np.squeeze(x.mean_), 'sigma':np.squeeze(x.std_)}))
        # fill the fully missing ones using this
        gs = StandardScalerDF(*ms.median())
        ssfits.loc[ms[ms.isna().any(axis=1)].index] = gs
        level = change_rates.groupby('country', group_keys=False)
        level = level.apply(lambda x: ssfits[x.name].inverse_transform(pd.DataFrame(x)).squeeze())
        level.name = actual.name
        return level

    @staticmethod
    def inv_change(change_rates, actual, method='ptc', freq_years=1):
        # define the growth rates relative to the first valid index for freq
        is_ptc = method == 'ptc'
        agg_method = 'prod' if is_ptc else 'sum'
        adj_method = 'div' if is_ptc else 'sub'
        base_value = 1 if is_ptc else 0
        window = int(freq_years*4)
        change_rates = change_rates + 1 if is_ptc else change_rates
        months = change_rates.index.get_level_values('date').month
        x = change_rates.groupby(['country', months]).expanding()
        x = x.apply(lambda x: getattr(x.iloc[::-freq_years], agg_method)()).reset_index(
            level=[0, 1], drop=True).sort_index()

        # define initial x values for reference growth dates
        xs = x.groupby('country').apply(lambda x: x[x.first_valid_index(
        ):].iloc[:window]).reset_index(level=0, drop=True)
        xs.index = xs.index.set_levels((xs.index.get_level_values('date') -
                                        pd.DateOffset(years=freq_years)).unique(),
                                       level=1)
        xs = pd.Series(base_value, index=xs.index).sort_index()
        x = pd.concat([x, xs]).sort_index()
        xs.index.names = ['country', 'ref_date']
        x.name = method
        x = x.reset_index()

        # construct and add reference growth dates to x
        ref_date = pd.Series(xs.index.get_level_values('ref_date').unique())
        ref_dates = [ref_date]
        actual_dates = xs.index.get_level_values('ref_date').unique()
        end_date = x['date'].max()
        while actual_dates.max() < end_date:
            actual_dates = actual_dates.union(actual_dates +
                                              pd.DateOffset(years=freq_years))
            ref_dates.append(ref_date)
        ref_dates = pd.concat(ref_dates)
        ref_dates = ref_dates.iloc[:x['country'].value_counts().iloc[0]]

        # adding the reference dates
        x['ref_date'] = pd.concat([ref_dates] * x['country'].nunique()).values
        x = x.set_index(['country', 'date'])

        # extract vaild actual values
        a = actual.groupby('country').apply(lambda x: x[x.first_valid_index(
        ):].iloc[:window]).reset_index(0, drop=True)

        # define monthly dataframe for the actual and vaild values
        am = pd.merge(a, x, 'left', ['country', 'date'])
        am['ref_level'] = getattr(am[a.name], adj_method)(am[method])
        am = am.drop(columns=[method, a.name])

        # fill in the fully missing values with the mode values for each date
        date_means = am.groupby('ref_date')['ref_level'].mean()
        am['ref_level'] = am['ref_level'].fillna(am['ref_date'].replace(date_means))

        # use the initial actual values and the cumulative growth rates to derive the entire series
        af = pd.merge(x, am, 'left', ['country', 'ref_date']).set_index('ref_date')
        af = pd.Series(getattr(af, agg_method)(axis=1).values, index=x.index)
        af.name = a.name
        return af

    def transform(self, filled_series, actual, method='ptc', freq_years=1, filter_threshold=0.95):
        if isinstance(method, type(None)):
            return filled_series
        elif method == 'norm':
            return self.inv_scale(filled_series, actual)
        window = int(freq_years*4)
        af = self.inv_change(filled_series, actual, method=method, freq_years=freq_years)

        # filter the filled series in case of high oscillation
        to_filter = self.oscillation_probability(af, actual, window=window) > filter_threshold
        if to_filter:
            af_missing = pd.merge(af, actual.isna().rename('missing'), 'left', on=['country', 'date'])
            af_missing = af_missing['missing'].fillna(True)
            aff = af.groupby('country').apply(self.filter_series,
                                              window=window).reset_index(level=0, drop=True)
            af.loc[af_missing] = np.NaN
            af = af.fillna(aff)
        return af


# In[16]:


def plot_qr_coeff(qcoef_table, variable, horizon=4, ax=None):
    variable_coeffs = qcoef_table.loc[horizon].loc[variable].copy()
    variable_coeffs = variable_coeffs.rename(columns={'coeff_scale': variable, 'coeff': variable})
    mean = variable_coeffs.loc['mean']
    variable_coeffs = variable_coeffs.drop('mean')
    variable_coeffs.index = variable_coeffs.index.astype(float)
    ax = variable_coeffs.plot(y=variable, title=f"{variable} horizon={horizon}", ax=ax)
    ax.fill_between('quantile', 'lower', 'upper', data=variable_coeffs.reset_index(), 
                    color='k', alpha=.1)
    ax.hlines(mean[variable], variable_coeffs.index.min(), variable_coeffs.index.max(), linestyle='--')
    ax.axhline(0, alpha=0.5)
    return ax


# In[17]:


def ridgeline_custom(X, Y, overlap=0.7, fill_color='y', fill=True, outline=True, line_color='k',
                     y_fontsize=10, y_interval=1, dist_density=None, inline=True):
    dates = X.index.unique()[::-1]
    if (dist_density) and (dist_density != 1):
        dates = pd.DatetimeIndex([x for i, x in enumerate(dates) if (i + 1) % dist_density != 0])
    ys = []
    for index, date in enumerate(dates):
        # Setting up inputs: PDF x and y
        xx = X.loc[date]
        yy = Y.loc[date]
        y = index*(1.0-overlap)
        zorder = len(dates)-index+1
        ys.append(y)

        # plotting each curve
        if inline:
            plt.plot(xx, yy+y, c=line_color, zorder=zorder)
        if fill:
            plt.fill_between(xx, np.ones(len(xx))*y, yy+y, zorder=zorder, color=fill_color)
        if outline:
            plt.plot(xx, yy+y, c=line_color, zorder=zorder)
    if isinstance(dates[0], pd.Timestamp):
        plt.yticks(ys[::y_interval], dates.strftime('%Y-%b')[::y_interval])
    else:
        plt.yticks(ys[::y_interval], dates[::y_interval], fontsize=y_fontsize)


# In[ ]:





# In[ ]:





# # Dataset cleanup

# ## Consistency based transformations

# In[127]:


# loading datasets
arm_data = pd.read_excel('../data/arm_data/armenia_data.xlsx', index_col=0)
panel_dataset = pd.read_csv('../data/panel_data/panel_dataset.csv')


# In[128]:


# setting up panel variables to add to the main model
add_columns_arm = [
    'CAR_T1', 'REER', 'Unemployment',
    'CPI', 'Lending_rate', 'Deposit_rate', 'Equity_index', 'Bond_yield_10Y',
    'Bond_yield_5Y', 'Bond_yield_2Y', 'Bond_yield_1Y', 'Tbill_rate',
    'MoneyMarket_rate', 'Gov_bond_rate', 'Stocks_return', 'Stocks_volatility',
    'Real_house_price', 'Rent_price', 'MP', 'MPI', 'MPI_capital',
]
panel_data_arm = panel_dataset[panel_dataset['country'] == 'Armenia'].set_index('date')
panel_data_arm = panel_data_arm.drop(columns='country')
panel_data_arm.index = pd.PeriodIndex(panel_data_arm.index, freq='Q').to_timestamp(freq='Q')
panel_data_arm = panel_data_arm[add_columns_arm]


# In[129]:


# combining available data
arm_data = pd.concat([arm_data, panel_data_arm], axis=1)


# In[130]:


# applying custom transformations
horizons = [4, 8, 12]
yearfreq = 4
target = 'Real GDP'

# adding GDP growth rate to the transformations set
Y = []
Ar = []
for horizon in horizons:
    x = arm_data[target].copy()
    annual_ar = ((x/x.shift(horizon))**(yearfreq/horizon)-1)*100
    annual_hz_gr = calc_growth_rate(x, horizon=horizon, yearfreq=yearfreq, method_growth='cpd')
    nyears = int(horizon/yearfreq)
    Ar.append(annual_ar.rename(f'autoregressive_{nyears}y'))
    Y.append(annual_hz_gr.rename(f'GDP_real_hz_{nyears}y'))
Ar = pd.concat(Ar, axis=1)
Y = pd.concat(Y, axis=1)


# In[131]:


X_raw = arm_data

# grouping columns based on origin
original_columns = list(X_raw.columns)
generated_sample = feature_engineering_pipeline.fit_transform(X_raw[[original_columns[0]]]).columns
col_lookup = re.compile(f"{original_columns[0]}", re.IGNORECASE)
generated_suffixes = [col_lookup.sub('', s) for s in generated_sample]
column_group_mapping = {c+s: c for s in generated_suffixes for c in original_columns}
column_groups = {c: [c+s for s in generated_suffixes] for c in original_columns}

# performing transformations
X_T = feature_engineering_pipeline.fit_transform(X_raw)


# In[132]:


index = 0
v_name = list(column_groups.keys())[index]
corr = pd.concat([Y, X_T[list(column_groups.values())[index]]], axis=1)
corr = corr.corr().abs().iloc[3:, :3]
corr_best = corr.reindex(corr.mean(axis=1).sort_values(ascending=False).index).head(40)
display(corr_best)
display(corr.loc[v_name])
X_T[[v_name] + list(corr_best.index[:6])].plot(subplots=True, layout=(1, 7), figsize=(30, 2))
plt.show()


# In[133]:


consistent_columns = [
    'Real GDP_PctC_1y', 'Financial Cycle Index', 'Level_log_PctC_1y',
    'Steepness_PctC_1y', 'Curvature', 'Policy rate_log_C_1y',
    'Bank RoE', 'Liquidity ratio (HLA/Assets)', 'CAR_C_1y',
    'Mortgage stock_PctC_3y', 'Mortgage flow_PctC_3y',
    'Mortgage interest rate_lag_1', 'Construction loans_PctC_2y',
    'Real estate prices_log', 'Exchange rate_lag_1',
    'Trade balance', 'Copper prices_lag_4_log', 'CAR_T1_lag_4',
    'REER', 'Unemployment_lag_4_C_3y', 'CPI_C_1y', 'Lending_rate_C_3y',
    'Deposit_rate_C_3y', 'Equity_index', 'Bond_yield_10Y_log_PctC_1y',
    'Bond_yield_5Y_log_PctC_3y', 'Bond_yield_2Y_log_PctC_3y',
    'Bond_yield_1Y_log_PctC_3y', 'Tbill_rate_log_PctC_2y',
    'MoneyMarket_rate_log_PctC_3y', 'Gov_bond_rate_lag_1_log_PctC_1y',
    'Stocks_return', 'Stocks_volatility_log_PctC_3y', 'Real_house_price',
    'Rent_price', 'MP', 'MPI', 'MPI_capital',
]
arm_data_consistent = X_T[consistent_columns].copy()


# ## Inverse transformation of consistently filled data

# In[134]:


imp_search_result = []
Xc = add_trend(arm_data_consistent, trend="ct")
Xc['trend2'] = Xc['trend']**2
Xc['trend3'] = Xc['trend']**3
Xc['trend4'] = Xc['trend']**4
for f in range(2, 38):
    imp = IterativeImputer(estimator=LinearRegression(fit_intercept=False), max_iter=1000, 
                           random_state=42, n_nearest_features=f)
    arm_data_filled = imp.fit_transform(Xc)
    arm_data_filled = pd.DataFrame(arm_data_filled, index=Xc.index, columns=Xc.columns)
    corr_dev = ((arm_data_filled.corr() - arm_data_consistent.corr())**2).sum().sum() / 2
    imp_search_result.append({'f': f, 'corr_dev': corr_dev})
imp_search_result = pd.DataFrame(imp_search_result).sort_values('corr_dev')

# rank the search results
fw = 0.4
rank = imp_search_result.reset_index(drop=True).reset_index()
rank['index'] = rank['index'] + 1
rank['fr'] = 38 - rank['f']
rank['score'] = rank['index']*(1-fw)+rank['fr']*fw
rank = rank.sort_values('score')
f_best = int(rank.iloc[0]['f'])

# perform the imputation
imp = IterativeImputer(estimator=LinearRegression(fit_intercept=False), max_iter=1000, 
                       random_state=42, n_nearest_features=f_best)
arm_data_filled = imp.fit_transform(Xc)
arm_data_filled = pd.DataFrame(arm_data_filled, index=Xc.index, columns=Xc.columns)
corr_dev = ((arm_data_filled.corr() - arm_data_consistent.corr())**2).sum().sum() / 2
print('Corr dev:', corr_dev)


# In[135]:


# selected transformation
c_transformations = {c: 'c' for c in ['_C_1y', '_C_2y', '_C_3y']}
ptc_transformations = {
    ptc: 'ptc' for ptc in ['_PctC_1y', '_PctC_2y', '_PctC_3y']
}
lag_transformations = {lag: 'lag' for lag in ['_lag_1', '_lag_2', '_lag_3', '_lag_4']}
transformations = {
    '_normalized': 'norm',
    '_log': 'log',
    **c_transformations,
    **ptc_transformations,
    **lag_transformations
}


# In[136]:


# inv_tf = InverseTransformer()
# arm_dataset = []
# for c in arm_data_filled.columns:
#     change = arm_data_filled[c]
#     actual = arm_data[column_group_mapping[c]]
#     tfs = [t for t in transformations if t in c]
#     for tf in tfs[::-1]:
#         freq = re.search(r'(\d{1,2})', tf)
#         freq = freq.group() if not isinstance(freq, type(None)) else freq
#         freq = int(float(freq)) if not isinstance(freq, type(None)) else freq
#         af = inv_tf.transform(change, actual, method=tf, freq=freq, filter_threshold=0.90)
#     arm_dataset.append(af)
# arm_dataset = pd.concat(arm_dataset, axis=1).sort_index()
# arm_dataset = arm_dataset.dropna()

# ToDo: apply all the transformations to actual, right before change
# then invert change to f(actual). Next repeat this until no tf left to apply to actual.
# thus acutal = inv(inv(change, f(actual)), actual).


# In[137]:


arm_dataset = arm_data_filled[arm_data_consistent.columns].iloc[12:].rename(columns=column_group_mapping)


# # Financial Conditions Index (FCI)

# In[138]:


# exog
exog_cols_fci = [
    'Level', 'Steepness', 'Curvature', 'Bank RoE', 'Mortgage interest rate',
    'Lending_rate', 'Deposit_rate', 'Equity_index',
    'Bond_yield_10Y', 'Bond_yield_5Y', 'Bond_yield_2Y', 'Bond_yield_1Y',
    'Tbill_rate', 'MoneyMarket_rate', 'Gov_bond_rate', 'Stocks_return',
    'Stocks_volatility'
]
X_fci = arm_dataset[exog_cols_fci].copy()

# endog
endog_cols = ['CPI', 'Real GDP', 'Policy rate']
Y_fci = arm_dataset[endog_cols].copy()


# In[139]:


# clustering the X_fci columns
sil_scores = get_silhouette_scores(X_fci.corr().abs(), n_init=10)
best_k = max(sil_scores, key=sil_scores.get)
kmeans = KMeans(n_clusters=best_k, n_init=10)
labels = kmeans.fit_predict(X_fci.corr().abs())
X_fci_col_order = np.argsort(labels)
X_fci_cm = {label: [X_fci.columns[i] for i, l in enumerate(labels) if l == label] for label in set(labels)}
sns.heatmap(X_fci.corr().abs().iloc[X_fci_col_order, X_fci_col_order], cbar=False, yticklabels=False, xticklabels=False)
plt.show()


# In[140]:


display(X_fci_cm)


# In[141]:


# standardize data
ss = StandardScalerDF()
X_fci_st = ss.fit_transform(X_fci)
Y_fci_st = ss.fit_transform(Y_fci)

# fillna with 0
X_fci_st = X_fci_st.fillna(0)
Y_fci_st = Y_fci_st.fillna(0)

# standardize again for some reason
X_fci_st = ss.fit_transform(X_fci_st)
Y_fci_st = ss.fit_transform(Y_fci_st)


# FCI using PCA

# In[142]:


pca_model = PCA(n_components=1, svd_solver='full', tol=0,
                iterated_power='auto', n_oversamples=10,
                power_iteration_normalizer='auto',
                random_state=42)
pca_model.fit(X_fci_st)
n = X_fci_st.shape[1]
pca_loadings = - pca_model.components_.T * np.sqrt(n)
fci_pca = (X_fci_st @ (pca_loadings / n)).squeeze().rename('fci_pca')
print('Explained Variance:', pca_model.explained_variance_ratio_[0].round(4))
print('Singular values:', pca_model.singular_values_[0].round(4))


# FCI using the method by Doz, Giannone and Reichlin (2011)

# In[143]:


t = Y_fci_st.shape[0]
YF = pd.concat([Y_fci_st, fci_pca], axis=1)
YX = pd.concat([Y_fci_st, X_fci_st], axis=1)
t, nvars = YX.shape
nlags = 4
nstates = YF.shape[1]
n_undefined_rows = nstates*(nlags-1)
nstates_doz = nstates*nlags


# In[144]:


# measurement model
measurement_model = sm.OLS(YX, YF).fit()
L = np.array(measurement_model.params)
e = measurement_model.resid
measurement_sigma = np.diag(np.diag((e.T @ e) / t))
R = measurement_sigma + 1e-10*np.eye(nvars)
# sigma2 = np.diag((e**2).sum() / t)

# state model
lf = LagFeatures(periods=list(range(1, nlags+1)), drop_original=True)
yf_lags = lf.fit_transform(YF).dropna()
yf_lags_cov_inv = np.linalg.inv(yf_lags.T @ yf_lags)
yf = YF.loc[yf_lags.index].copy()
t = yf.shape[0]
state_model = sm.OLS(yf, yf_lags).fit()
T = np.array(state_model.params)
e = state_model.resid
sigmaf = (e.T @ e) / (t-1)
state_beta_var = np.kron(sigmaf, yf_lags_cov_inv)
bb = T.reshape(nstates, nlags, nstates).transpose(0, 2, 1).reshape(-1)

# state model as VAR
ar_identity = np.concatenate((np.eye(n_undefined_rows), np.zeros((n_undefined_rows, nstates))), axis=1)
T_doz = np.concatenate((T.T, ar_identity), axis=0)
Q_doz = np.concatenate((sigmaf, np.zeros((nstates, n_undefined_rows))), axis=1)
Q_doz = np.concatenate((Q_doz, np.zeros((n_undefined_rows, nstates_doz))), axis=0)

# drawing states from Kalman filter (smoothing)
S0 = np.zeros(nstates_doz)
P0 = 10*np.eye(nstates_doz)
S_draws = kalman_companion(YX, S0, P0, L, R, T_doz, Q_doz)
S_draws = pd.DataFrame(S_draws, index=YF.index, columns=YF.columns).rename(columns={'fci_pca': 'fci_doz'})
fci_doz = S_draws['fci_doz']


# FCI using the method by Koop and Korobilis (2013)

# In[145]:


# step 1: Update Parameters Conditional on PC
# the  decays include:
# 1) decay factor for measurement error variance
# 2) decay factor for factor error variance
# 3) decay factor for loadings error variance
# 4) decay factor for VAR coefficients error variance
decays = np.array([0.96, 0.96, 0.99, 0.99])
nfactors = 1
nlags = 4
nstates = YF.shape[1]
nmacro = nstates - nfactors
t, nvars = YX.shape
nstates_all = nstates*nlags
nVAR = nlags*(nstates**2)  # number of VAR parameters

# Priors
priors = {}

# initial condition on the factors
priors['factor_0_mean'] = np.zeros(nstates_all)
priors['factor_0_var'] = 10*np.eye(nstates_all)

# initial condition on lambda_t
priors['lambda_0_mean'] = np.zeros((nvars, nstates))
priors['lambda_0_var'] = 1*np.eye(nstates)

# initial condition on beta_t
gamma = 0.1
b_prior, Vb_prior = Minn_prior_KOOP(gamma, nstates, nlags, nVAR)  # Obtain a Minnesota-type prior
priors['beta_0_mean'] = b_prior
priors['beta_0_var'] = Vb_prior

# initial condition on the covariance matrices
priors['V_0'] = 0.1*np.eye(nvars)
priors['V_0'][:nmacro, :nmacro] = 0
priors['Q_0'] = 0.1*np.eye(nstates)


# In[146]:


tvp_favar_korobilis = TVP_FAVAR(YX, YF, nlags)
beta_t, beta_new, lambda_t, V_t, Q_t = tvp_favar_korobilis.fit(nfactors, decays, priors)
factor_new, Sf_t_new = tvp_favar_korobilis.gen_factors(priors)
S_knk = pd.DataFrame(factor_new, index=YF.index, columns=YF.columns).rename(columns={'fci_pca': 'fci_knk'})
fci_knk = S_knk['fci_knk'].copy()


# In[147]:


fig, ax = plt.subplots(figsize=(12, 5))
ss.fit_transform(fci_pca).plot(ax=ax)
ss.fit_transform(fci_doz).plot(ax=ax)
ss.fit_transform(fci_knk).plot(ax=ax)
plt.ylabel("FCI")
plt.legend()
plt.savefig('../results/FCI-Armenia.png', bbox_inches='tight', dpi=300)
plt.show()


# In[148]:


arm_dataset['Financial Conditions Index'] = fci_knk


# # Dataset overview

# In[149]:


col_order = [
    'Real GDP', 'Financial Cycle Index', 'Financial Conditions Index',
    'Policy rate', 'MPI', 'CAR', 'CAR_T1', 'Bank RoE',
    'Liquidity ratio (HLA/Assets)', 'Level', 'Steepness', 'Curvature',
    'Lending_rate', 'Deposit_rate', 'Equity_index', 'Bond_yield_10Y',
    'Bond_yield_5Y', 'Bond_yield_2Y', 'Bond_yield_1Y', 'Tbill_rate',
    'MoneyMarket_rate', 'Gov_bond_rate', 'Stocks_return',
    'Stocks_volatility', 'CPI', 'Exchange rate',
    'Trade balance', 'Copper prices', 'REER', 'Unemployment',
    'Mortgage stock', 'Mortgage flow', 'Mortgage interest rate',
    'Construction loans', 'Real estate prices', 'Real_house_price',
    'Rent_price', 'MP', 'MPI_capital'
]
# reorder and drop columns from FCI
arm_dataset = arm_dataset[col_order].copy()
arm_dataset = arm_dataset.drop(columns=exog_cols_fci)


# In[150]:


display(arm_dataset.describe().T.round(1))


# In[151]:


corr = arm_dataset.corr()
corr_target = corr.iloc[0]
corr_target = corr_target.loc[corr_target.abs().sort_values(ascending=False).index]
display(corr_target)


# In[152]:


# # assigning a color for each column
# dict_groups = {
#     'autoregressive': ['real_y_ms_yoy']
# }
# color_mapping = {'autoregressive':'#1f77b4', 'financial_conditions':'#ff7f0e', 
#                  'policy_conditions':'#2ca02c', 'intermediation':'#d62728',
#                 'housing':'#9467bd', 'leverage':'#8c564b', 'external_sector':'#e377c2',
#                  'real_y_ms':'purple'
#                 }
# node_colors = {value: color_mapping[key] for key, values in (dict_groups).items() for value in values}

# drop_cols = []
# working_corr = corr.drop(index=drop_cols, columns=drop_cols).copy()
# plot_minimum_spanning_tree(working_corr, title="Macro variables - Minimum Spanning Tree",
#                            threshold=0.55, node_size=50, 
#                            color_legend=color_mapping, node_color=node_colors,
#                            width=1)
# plt.show()


# In[153]:


# Compute the correlation matrix
corr = arm_dataset.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"fontsize": 7})
plt.show()


# In[154]:


arm_overview = pd.concat([Y, arm_dataset.iloc[:, 1:]], axis=1)
corr = arm_overview.corr()
corr_target = corr.iloc[0]
corr_target = corr_target.loc[corr_target.abs().sort_values(ascending=False).index]
display(corr_target)


# In[155]:


# Compute the correlation matrix
corr = arm_overview.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"fontsize": 7})
plt.show()


# # Modeling

# In[179]:


# setup
sns.reset_defaults()
quantlist = np.arange(0.05, 1, 0.05).round(2)
model_drop_cols = ['Real GDP', 'CAR_T1', 'Exchange rate',
                   'Mortgage stock', 'MP', 'MPI_capital',
                   'Real estate prices']
exog = arm_dataset.iloc[1:].drop(columns=model_drop_cols)


# In[180]:


# setup dataset
macro_dataset = []
for index, y_name in enumerate(Y):
    horizon = int(float(y_name[-2])*4)
    endog = Y[y_name].copy()
    ar = Ar.iloc[:, index].copy()
    macro_data = pd.concat([ar.rename(ar.name[:-3]), exog], axis=1)
    macro_data = pd.concat([endog.rename(endog.name[:-6]), macro_data], axis=1)
    macro_data['horizon'] = horizon

    # remove extra rows
    cutoff_date = macro_data.isna().groupby('date').sum().sum(axis=1).idxmin()
    macro_data = macro_data[macro_data.index.get_level_values('date') >= cutoff_date].copy()

    # add to the dataset
    macro_dataset.append(macro_data)
macro_dataset = pd.concat(macro_dataset).reset_index().set_index(['horizon', 'date'])


# In[181]:


# fitting QR the model
arm_qcoef = []
arm_qcoef_norm = []
arm_qr_pred = []

for horizon, macro_data in macro_dataset.groupby('horizon'):
    target = macro_data.columns[0]
    macro_data = macro_data.reset_index('horizon', drop=True).reset_index()
    quantfit_out = run_quantfit(macro_data, target=target, horizon=horizon,
                                model_fit_args={'max_iter': 20000, 'p_tol': 1e-06},
                                quantlist=quantlist)
    arm_qcoef.append(quantfit_out['localprj'])
    arm_qcoef_norm.append(quantfit_out['qcoef'])
    arm_qr_pred.append(quantfit_out['cond_quant'])
arm_qcoef = pd.concat(arm_qcoef).set_index(['horizon', 'variable', 'quantile'])
arm_qcoef_norm = pd.concat(arm_qcoef_norm).set_index(['horizon', 'variable', 'quantile'])
arm_qr_pred = pd.concat(arm_qr_pred).reset_index().set_index(['horizon', 'tau', 'date'])
