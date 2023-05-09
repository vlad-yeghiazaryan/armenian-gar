#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[41]:


# importing main libs
import numpy as np
import pandas as pd

# libs for plots
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as matcolors
from matplotlib.colors import LightSource
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# utils
from dateutil import parser
import pprint as pp

# models
import statsmodels.api as sm

# models (import)
from GAR.partition import retropolated_PCA
from GAR.quantfit import run_quantfit
from GAR.tsfit import run_tsfit, get_cond_quant, select_df_partition
from GAR.scenario import run_scenario, gen_shocked_PCA
from GAR.historical import run_historical
from GAR.segment import run_segment

# # params
# sns.set(style='whitegrid')
# sns.set(rc={'figure.figsize':(4,4)})


# In[2]:


# Setting up parameters

# Partition groups
autoregressive = ['real_y_ms_yoy']
financial_conditions = ['consumer_AMD', 'consumer_USD', 
                        'mortgage_AMD', 'mortgage_USD', 
                        'business_AMD', 'business_USD']
policy_conditions = ['tbill', 'gbond','policy_rate']
intermediation = ['loan_to_deposit', 'spread(loan-deposit)', 'spread(loan-tbill)']
housing = ['mortgage_stock_yoy', 'rep_yoy', 'rep_deviation']
leverage = ['credit_to_gdp_gap', 'household_credit_stock_yoy', 'business_credit_stock_yoy', 
            'npl', 'assets_equity']
external_sector = ['exchange_rate', 'net_tr_yoy', 'rus_y_yoy', 'copper_yoy_lag2', 'copper_yoy']

# global parmas
dict_global_params = {'target': 'real_y_ms', 'horizon':4}
dict_groups = {
    'autoregressive': autoregressive, 
    'financial_conditions': financial_conditions, 
    'policy_conditions':policy_conditions,
    'intermediation': intermediation, 
    'housing': housing,
    'leverage': leverage,
    'external_sector':external_sector
}

# assigning a color for each group
color_mapping = {
    'autoregressive':'#1f77b4', 'financial_conditions':'#ff7f0e', 
    'policy_conditions':'#2ca02c', 'intermediation':'#d62728',
    'housing':'#9467bd', 'leverage':'#8c564b', 'external_sector':'#e377c2',
    'real_y_ms':'purple'
}
target_group = {
    'real_y_ms':['real_y_ms']
}
node_colors = {value: color_mapping[key] for key, values in (dict_groups|target_group).items() for value in values}


# In[3]:


# importing data and perform pre-processing
df_partition = pd.read_excel('../data/gar_main.xlsm', 'Data')

# interpolate and forward fill missing values
for column in df_partition:
    c = df_partition[column].copy()
    if c.dtype not in [float, np.float64, np.float32]:
        continue
    f_index = c.first_valid_index()
    l_index = c.last_valid_index()
    c.loc[:l_index] = c.loc[:l_index].interpolate(method='linear')
    c.loc[f_index:] = c.loc[f_index:].fillna(method='ffill', limit=None)
    df_partition[column] = c


# In[4]:


# imf partition output
imf_data = pd.read_excel('../data/gar_V2.xlsm', 'Output_partitions')
imf_data.dropna(axis=0, how='any', inplace=True)
imf_data.set_index('date', inplace=True)


# # Partitioning the input data

# In[5]:


choice_columns = [
    'autoregressive', 'policy_conditions', 'intermediation',
    'housing_diff', 'external_sector_diff', 'leverage_L4',
    'external_sector_L1'
]

t_mapping = {
    **{v:v for v in choice_columns},
    'intermediation':'intermediation', 
    'intermediation_detrend':'intermediation_detrend',
    'financial_conditions':'financial_conditions',
    'financial_conditions_diff':'financial_conditions_diff',
    'housing':'housing',
    'housing_diff':'housing_diff',
    'external_sector':'external_sector',
    'external_sector_diff':'external_sector_diff',
    'external_sector_L1':'external_sector_L1',
    'leverage':'leverage',
    'leverage_L4':'leverage_L4'
}

def polytrend(series, level=1):
    index = series.reset_index().index
    z = np.polyfit(index, series, 2)
    p = np.poly1d(z)
    trend = p(index)
    return trend

# MVA
# partition[self.mapping['domestic_macro_MVA']] = partition[self.mapping['domestic_macro']].rolling(window=2).mean()
# partition.drop(columns=self.mapping['domestic_macro'], inplace=True)

class DataTransformer():
    def __init__(self):
        self.choice_columns = choice_columns
        self.mapping = t_mapping
    def transform(self, partition, depvar):
        
        # Lagged
        partition[self.mapping['leverage_L4']] = partition[self.mapping['leverage']].shift(4)
        partition[self.mapping['external_sector_L1']] = partition[self.mapping['external_sector']].shift(1)
        
        # Power
        # partition[reg_long] = partition[reg_short]**(n)
        
        # Diff
        partition[self.mapping['financial_conditions_diff']] = partition[self.mapping['financial_conditions']].diff(4)
        partition[self.mapping['housing_diff']] = partition[self.mapping['housing']].diff(4)
        partition[self.mapping['external_sector_diff']] = partition[self.mapping['external_sector']].diff(4)
        
        # Trend
        trend = polytrend(partition[self.mapping['intermediation']], 2)
        partition[self.mapping['intermediation_detrend']] = partition[self.mapping['intermediation']] - trend
        
         # ChangeRate
        # partition[reg_long] = partition[reg_short].pct_change(n)
        
        # Remove columns not in choice
        selected_columns = [value for key, value in self.mapping.items() if key in self.choice_columns]
        partition = partition[['date', depvar] + selected_columns].copy()
        return partition

# Lets try the transformer
depvar  = dict_global_params['target'] + '_hz_' + str(dict_global_params['horizon'])
transformer = DataTransformer()
macro_data, partition_load, partition_log = retropolated_PCA(df_partition, dict_groups, **dict_global_params)
macro_data = transformer.transform(macro_data, depvar)
# display(macro_data)


# In[ ]:





# # Quantile regression fit

# In[6]:


# quantile regression fit step (2)
quantlist = [0.1, 0.25, 0.5, 0.75, 0.9]
dict_output_quantfit = run_quantfit(macro_data, quantlist=quantlist,
                                    model=sm.QuantReg, **dict_global_params)


# In[7]:


qcoef = dict_output_quantfit['qcoef'].reset_index(drop=True)
cond_quant_series = dict_output_quantfit['cond_quant'].reset_index()
cond_quant_series.head()


# # T-skew fit

# In[8]:


# tsfit input data
# mode: Free vs Fixed
# fittype: 'T-skew' vs "Asymmetric T"
latest_date = parser.parse('2020-03-31')
fit_params = {
    'fittype': 'T-skew',
    'mode': {'constraint': 'Free', 'value':None},
    'qsmooth': 'None',
    'qsmooth_period':2,
    'plot_mode': True,
    'plot_median': True,
    'plot_mean': False,
    'dof': {'constraint': 'Default', 'value': None},
    'var_low': {'constraint': 'Default', 'value': None},
    'var_high': {'constraint': 'Default', 'value': None},
    'skew_low': {'constraint': 'Default', 'value': None},
    'skew_high': {'constraint': 'Default', 'value': None}
}
# tsfit step (3)
dict_output_tsfit = run_tsfit(latest_date, fit_params, macro_data, qcoef, **dict_global_params)
dfpdf = dict_output_tsfit['dfpdf']
fitted_params = dict(dict_output_tsfit['result'])
print(dfpdf)


# In[9]:


dict_output_tsfit['fig']


# # Counterfactual Scenarios Analysis

# In[10]:


# scenario input data
latest_date = parser.parse('2020-03-31')
fit_params = {
    'fittype': 'T-skew',
    'mode': {'constraint': 'Free', 'value':None},
    'dof': {'constraint': 'Default', 'value': None},
    'var_low': {'constraint': 'Default', 'value': None},
    'var_high': {'constraint': 'Default', 'value': None},
    'skew_low': {'constraint': 'Default', 'value': None},
    'skew_high': {'constraint': 'Default', 'value': None}
}

fit_params_shocked = {
    'fittype': 'T-skew',
    'mode': {'constraint': 'Free', 'value':None},
    'dof': {'constraint': 'Default', 'value': None},
    'var_low': {'constraint': 'Default', 'value': None},
    'var_high': {'constraint': 'Default', 'value': None},
    'skew_low': {'constraint': 'Default', 'value': None},
    'skew_high': {'constraint': 'Default', 'value': None}
}
# shocktype: 'By +/- STD' or 'By +/- percentage' 
shockvars = {
    'policy_rate': {
        'shocktype': 'By +/- STD',
        'shockvalue': 10
    }
}
# generating shocked series
transformer = DataTransformer()
df_shockedvar, df_shockedgrp = gen_shocked_PCA(shockvars, dict_groups, df_partition, 
                                               transformer, **dict_global_params)
df_shockedgrp.columns = df_shockedgrp.columns.str.replace('_shocked', '')

# calculating conditional quantiles
cond_quant = get_cond_quant(latest_date, macro_data, qcoef, **dict_global_params)
cond_quant_shocked = get_cond_quant(latest_date, df_shockedgrp, qcoef, **dict_global_params)

dict_output_scenario = run_scenario(latest_date, cond_quant, cond_quant_shocked, 
                                    fit_params, fit_params_shocked, **dict_global_params)


# In[11]:


df_shockedvar.loc[latest_date]


# In[12]:


print('Results')
print(pd.DataFrame(dict_output_scenario['res']))


# In[13]:


dict_output_scenario['fig']


# # Historical dist

# In[168]:


# mode: Free vs Fixed
# fittype: 'T-skew' vs "Asymmetric T"
hist_fit_params = {
    'fittype': 'T-skew',
    'mode': {'constraint': 'Free', 'value':None},
    'qsmooth': 'None',
    'qsmooth_period':2,
    'dof': {'constraint': 'Default', 'value': None},
    'var_low': {'constraint': 'Default', 'value': None},
    'var_high': {'constraint': 'Default', 'value': None},
    'skew_low': {'constraint': 'Default', 'value': None},
    'skew_high': {'constraint': 'Default', 'value': None}
}

dict_input_historical = {
    'start_date': parser.parse('2003-03-31'),
    'end_date': parser.parse('2022-03-31'),
    'time_inc': 1,
    'fit_params': hist_fit_params,
    **dict_global_params
}
dict_output_historical = run_historical(dict_input_historical, macro_data, qcoef)

# In[169]:


hist_sim_data = dict_output_historical['data']
hist_sim_data.head()


# In[175]:


# dict_output_historical['figs']['res']


# In[176]:


dict_output_historical['charts'][-1]


# # Multiple Horizon Projections

# In[18]:


# qsmooth_period: int or "auto" (in case of auto its equal to the horizon)
segment_fit_params = {
    'fittype': 'T-skew',
    'mode': {},
    'qsmooth': "None",
    'qsmooth_period': 2,
    'dof': {'constraint': 'Default', 'value': None},
    'var_low': {'constraint': 'Default', 'value': None},
    'var_high': {'constraint': 'Default', 'value': None},
    'skew_low': {'constraint': 'Default', 'value': None},
    'skew_high': {'constraint': 'Default', 'value': None}
}

dict_input_segment = {
    'horizonlist': [4, 8, 12, 16],
    'method_growth':'cpd',
    'retropolate':'Yes',
    'partition_groups': dict_groups,
    'transformer':transformer,
    'quantlist':[0.1, 0.25, 0.5, 0.75, 0.9],
    'fitdate': parser.parse('6/30/2018'),
    'fit_params': segment_fit_params,
    'fitconstrainlist': ['Free']*4,
    'fitconstrainvalues': [None]*4,
    **dict_global_params
}
dict_output_segment = run_segment(dict_input_segment, df_partition, model=sm.QuantReg)


# In[19]:


pd.DataFrame(dict_output_segment['res'])


# In[20]:


dict_output_segment['tails']


# In[21]:


dict_output_segment['dfpdf'].head()


# In[22]:


dict_output_segment['fig']


# In[23]:


# df_term['error_hz'+str(horizon)]=(df_quantcoef['upper']-df_quantcoef['lower'])/2
# loc=tsfit['loc']/tsfit['scale']
# loc=tsfit['loc']


# # Results

# In[160]:


qfit_series = cond_quant_series.pivot_table(values='conditional_quantile_mean', index='date', columns='tau')
qfit_series = pd.merge(qfit_series, macro_data[['date','real_y_ms_hz_4']], how='left', on='date')
qfit_series = qfit_series[[qfit_series.columns[-1]] + list(qfit_series.columns[:-1])]
qfit_series = qfit_series.set_index('date')
palette = {
    0.1: 'red',
    0.25: 'green',
    0.5: 'orange',
    0.75: 'purple',
    0.9: 'brown',
    'mean': 'gray',
    'real_y_ms_hz_4': 'black'
}
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=qfit_series, palette=palette, ax=ax)
plt.title('Quantile regression fit over time for all values')
plt.ylabel('GDP growth')
plt.show()


# In[161]:


qdist = qfit_series.reset_index().melt(id_vars='date', value_vars=['real_y_ms_hz_4',*quantlist, 'mean'])
fig, ax = plt.subplots(figsize=(12, 5))

sns.scatterplot(data=qdist, x='date', y='value', hue='date', style='variable', legend=True, ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-8:], labels[-8:], frameon=True)
plt.title('Quantile regression fit over time for all values')
plt.ylabel('GDP growth') 
plt.show()


# In[173]:


# custom outline
sns.set_theme(style="white", palette='viridis')

# Data preparation
# Tskew vs AsymT
method = 'Tskew'
df_PDF = pd.concat(hist_sim_data['dfpdf'].values, keys=hist_sim_data['dfpdf'].index).reset_index(level=1, drop=True)
df_PDF = df_PDF.reset_index().rename(columns={'index':'date'})
gdp_growth = df_PDF.groupby('date')[method+'_PDF_x'].apply(lambda x: list(x)).apply(lambda x: pd.Series(x))
gdp_density = df_PDF.groupby('date')[method+'_PDF_y'].apply(lambda x: list(x)).apply(lambda x: pd.Series(x))
dates = gdp_growth.index
dates_grid = np.repeat(dates, len(gdp_growth.columns)).to_numpy().reshape(-1, len(gdp_growth.columns))
dates_grid_values = np.repeat(np.arange(len(dates)), len(gdp_growth.columns)).reshape(-1, len(gdp_growth.columns))

# 3D plotting
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(dates_grid_values, gdp_growth, gdp_density, cmap=cm.viridis, 
                norm=matcolors.SymLogNorm(linthresh=0.02, linscale=0.03))
ax.set_xticks(np.arange(len(dates))[0::13])
ax.set_xticklabels(dates[0::13].strftime('%Y-%b'), fontsize=8)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('GDP growth', fontsize=10)
ax.set_zlabel('Density', fontsize=10)
plt.show()


# In[174]:


sns.reset_defaults()
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

fig, ax = plt.subplots(1, 1)
ax.pcolormesh(dates_grid_values, gdp_growth, gdp_density, cmap=cm.viridis,
              norm=matcolors.SymLogNorm(linthresh=0.02, linscale=0.03))
ax.set_xticks(np.arange(len(dates))[0::10])
ax.set_xticklabels(dates[0::10].strftime('%Y-%b'), fontsize=8)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('GDP growth', fontsize=10)
plt.title('GDP distribution over time (color indicates density)')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




