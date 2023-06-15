   
## 3rd-party modules
import numpy as np
import pandas as pd
from datetime import datetime as date 

## Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from statsmodels.api import QuantReg
import statsmodels.api as sm
from scipy.optimize import minimize, root
from scipy.stats import t, norm
from scipy.special import gamma
from scipy import interpolate, pi

def cond_quant_fits(dates, data, qcoeff, fit_params, methods):
    horizon = qcoeff['horizon'].iloc[0]
    scenario = qcoeff['scenario'].iloc[0]
    cond_quants, olsmeans = get_cond_quants(dates, data, qcoeff, 
                                            fit_params['qsmooth'], fit_params['qsmooth_period'])
    res = []
    for method in methods:
        # Update the fitparam mode before every fit iteration
        fit_params.update({'fittype':method})
        for fitdate, cond_quant in cond_quants.items():
            olsmean = olsmeans.loc[fitdate]
            model_fit, modx = dist_model_fit(cond_quant, fit_params, olsmean)
            res_fit = {
                'date':fitdate,
                'cond_quant': cond_quant,
                'model_fit': model_fit,
                'modx': modx,
                'olsmean':olsmean,
                'method':method,
                'scenario':scenario,
                'horizon':horizon
            }
            res.append(res_fit)
    return res

def dist_model_fit(cond_quant, fitparam, olsmean):
    # T-skew fit
    if fitparam['fittype']=='T-skew':
        model_fit = tskew_fit(cond_quant, fitparam)
        modx = model_fit['loc']
        
    # Asymmetric T-skew fit
    elif fitparam['fittype']=='Asymmetric T':
        model_fit = asymt_fit(cond_quant, fitparam, olsmean)
        modx = model_fit['loc']

    # Kernel fit
    elif fitparam['fittype']=='Kernel-based':
        cond_quant_uncross = quantile_uncrossing(cond_quant)
        model_fit = Weighted_kernel(cond_quant_uncross, bandwidth=fitparam['mode']['bandwidth'])
        model_fit.w_kernel_fit()

        # getting the mode
        x = model_fit.q_values
        ypdf = model_fit.w_kernel_pdf(x)
        modx = x[np.argmax(ypdf)]
    return model_fit, modx

def get_cond_quants(dates, data, qcoef, qsmooth='None', qsmooth_per=2):
    # selected the dates for estimating conditional quantiles
    cond_quants=[]
    olsmeans=[]
    for fitdate in dates:
        cond_quant = get_cond_quant(fitdate, data, qcoef, qsmooth=qsmooth, qsmooth_per=qsmooth_per)
        olsmean = cond_quant.pop('mean')

        # skip in cases where fitted values for the dates are missing
        if pd.isna(olsmean):
            continue
        olsmeans.append(olsmean)
        cond_quants.append(cond_quant)
    cond_quants = pd.DataFrame(cond_quants, index=dates).T.to_dict()
    olsmeans = pd.Series(olsmeans, index=dates)
    return cond_quants, olsmeans

def get_cond_quant(fitdate, data, qcoef, qsmooth='None', qsmooth_per=2):
    df_partition_fit = select_df_partition(fitdate, data, qsmooth, qsmooth_per)
    cond_quant = qcoef.groupby('quantile').apply(lambda x: df_partition_fit @ x.set_index('variable')['coeff_noscale'].sort_index())
    return cond_quant.to_dict()

def select_df_partition(fitdate, df, qsmooth='None', qsmooth_per=2):
    # Fitdat
    per = 0 if qsmooth_per==None else int(qsmooth_per)
    df = df.copy()
    if qsmooth=='None':
        df_partition_fit = df.loc[fitdate]
                           
    elif qsmooth=='Median':
        df_partition_fit = df[df.index<=fitdate].tail(per).median()
        
    elif qsmooth=='Mean':        
        df_partition_fit = df[df.index<=fitdate].tail(per).mean()
    else:
        df_partition_fit = df.loc[fitdate]
    
    df_partition_fit['const'] = 1
    df_partition_fit = df_partition_fit.sort_index()
    return df_partition_fit

def qRegFit(y, scenario, scenario_data, quantlist, horizon, model=QuantReg):
    regressors = scenario_data.columns
    depvar = y.name
    df_quantfit = pd.merge(y, scenario_data, how='outer', on='date')
    qcoeff, cond_quant, local_prj, exitcode = condquant(df_quantfit, depvar, regressors, horizon, quantlist, model)
    qcoeff['horizon'] = horizon
    cond_quant['horizon'] = horizon
    qcoeff['scenario'] = scenario
    cond_quant['scenario'] = scenario
    return qcoeff, cond_quant

def condquant(dall, depvar, regressors_avl, horizon, ql, model):   
    ql.sort()
    
    dall=dall.dropna(subset=regressors_avl)
    qrs = QuantileReg(depvar, indvars=regressors_avl,
                      quantile_list=ql,
                      data=dall,
                      scaling=True, alpha=0.1)

    dc = qrs.coeff
    dc.insert(0, 'variable', dc.index)
        
        ## Without scaling: get the conditional quantiles 
    qru = QuantileReg(depvar, indvars=regressors_avl,
                      quantile_list=ql, data=dall,
                      model=model,
                      scaling=False, alpha=0.1)

        ## Run the predictions on the full frame (estimates can differ)
    dcq = qru.cond_quant
    
    ## Store the coefficients
    dci = qru.coeff
    dci.insert(0, 'variable', dci.index)
    dc.rename(columns={'coeff':'coeff_scale'},inplace=True)
    dc['coeff_noscale']=dci['coeff']
    dc=dc[['variable','quantile','coeff_scale','coeff_noscale','pval','lower','upper','R2_in_sample','normalized', 'Model']]

    exitcode=1
    return [dc,dcq,dci,exitcode]

# Run the quantiles regressions
class QuantileReg(object):
    """ 
    Fit a conditional regression model, via quantile regressions

    Inputs:
    - depvar: string, dependent variable 
    - indvars: list of independent variables
    - quantile_list: list of quantiles to run the fit on
    - data = data to train the model on
    - scaling: zscore of the variables: standardized coefficients
    - alpha: the level of confidence to compute the confidence intervals
    
    Output:
    - qfit_dict = regressions fit, per quantiles (statsmodels object)
    - mfit = OLS regression fit, for the conditional mean
    - coeff = coefficients of the quantile regression, for every quantile
    - cond_quant: conditional quantiles and mean 

    Usage:
    qr = QuantileReg('y_growth_fwd_4', indvars=p_indvars, quantile_list=ql,
                     data=df, scaling=True, alpha=0.2)

    """
    __description = "Conditional quantiles, based on quantile regressions"
    __author = "Romain Lafarguette, IMF/MCM, rlafarguette@imf.org"

    ## Initializer
    def __init__(self, depvar, indvars, quantile_list, data, model=QuantReg, scaling=True, alpha=0.1):

        ## Parameters
        self.scaling = scaling
        self.alpha = alpha
        self.quantile_list = quantile_list
        
        ## Variables
        self.depvar = depvar

        ## Data cleaning for the regression
        self.data = data.dropna(subset=[self.depvar], axis='index', how='any').copy()

        # Model to use for quantile fitting
        self.QModel = model
        
        ## List of regressors
        self.regressors = [x for x in indvars if x in self.data.columns]
        
        ## Depending on user input, scale the variables
        vars_reg = [self.depvar] + self.regressors
        if self.scaling == True:
            self.data.loc[:, vars_reg] = scale(self.data.loc[:, vars_reg])
        else:
            pass
        
        ## From class methods (see below)
        self.qfit_dict = self.__qfit_dict()
        self.mfit = self.__mfit()
        self.coeff = self.__coeff()

        ## Conditional quantiles: use as predictors the historical regressors
        ## Basically, in-sample prediction but can be customized
        self.cond_quant = self.cond_quantiles(predictors=data)

    def __qfit_dict(self): 
        """ Estimate the fit for every quantiles """
        qfit_dict = dict()
        for tau in self.quantile_list:
            y = self.data[self.depvar]
            X = sm.add_constant(self.data[self.regressors])
            qfit = self.QModel(y, X).fit(q=tau, max_iter=4000, p_tol=1e-05)

            qfit_dict[tau] = qfit
        return(qfit_dict)

    def __mfit(self): 
        """ Estimate the fit for every quantiles """
        y = self.data[self.depvar]
        X = sm.add_constant(self.data[self.regressors])
        mfit = sm.OLS(y, X).fit()
        return(mfit)

    def __coeff(self):
        """ Extract the parameters and package them into pandas dataframe """
        params = pd.DataFrame()
        for tau in self.quantile_list:
            qfit = self.qfit_dict[tau]
            stats = [qfit.params,qfit.pvalues,qfit.conf_int(alpha=self.alpha)]
            stats_names = ['coeff', 'pval', 'lower', 'upper']
            dp = pd.concat(stats, axis=1); dp.columns = stats_names
            dp.insert(0, 'quantile', qfit.q) # Insert as a first column
            dp['R2_in_sample'] = qfit.prsquared
            dp['Model'] = qfit
            ## Add the scaling information
            dp.loc[:,'normalized'] = self.scaling
            params = pd.concat([params, dp])

        ## For information,  coeffs from an OLS regression (conditional mean)
        mfit = self.mfit
        stats = [mfit.params, mfit.pvalues, mfit.conf_int(alpha=self.alpha)]
        stats_names = ['coeff', 'pval', 'lower', 'upper']
        dmp = pd.concat(stats, axis=1); dmp.columns = stats_names
        dmp.insert(0, 'quantile', 'mean') # Insert as a first column
        dmp['R2_in_sample'] = mfit.rsquared
        dmp['Model'] = mfit
        ## Add the scaling information
        dmp.loc[:,'normalized'] = self.scaling
        coeff = pd.concat([params, dmp], axis='index')
        
        ## Return the full frame
        return(coeff)
    
    def cond_quantiles(self, predictors):
        """ 
        Estimate the conditional quantiles in sample 
        - Predictors have to be a pandas dataframe with regressors as columns
        """
        cond_quantiles = pd.DataFrame()
        X = sm.add_constant(predictors[self.regressors])
                
        for tau in self.quantile_list:
            qfit = self.qfit_dict[tau]
            # Run the prediction over a predictors frame     
            dc = qfit.get_prediction(exog=X).summary_frame()
            dc.columns = ['conditional_quantile_' + x for x in dc.columns]    
            ## Insert extra information            
            dc.insert(0, 'tau', tau)
            dc = dc.set_index(predictors.index)
            dc.insert(1, 'realized_value', predictors.loc[:, self.depvar])    
            cond_quantiles = pd.concat([cond_quantiles, dc])
                        
        ## Add the conditional mean
        dm = self.mfit.get_prediction(exog=X).summary_frame()
        dm.columns = ['conditional_quantile_' + x for x in dm.columns]    
        dm.insert(0, 'tau', 'mean')
        dm = dm.set_index(predictors.index)

        ## Insert the realized value (depvar is y(t+h))
        dm.insert(1, 'realized_value', predictors.loc[:, self.depvar])
        
        ## Concatenate both frames
        cq = pd.concat([cond_quantiles, dm])

        return(cq)

def growth_horizons(target, horizonlist, original_data, method_growth='cpd'):
    y = original_data.set_index('date')[target]
    y_horizons = []
    for horizon in horizonlist:
        if method_growth=='cpd':
            y_growth = cum_gr(y, horizon)
        elif method_growth=='yoy':
            y_growth = yoy_gr(y, horizon)
        y_growth.name = target + '_hz_' + str(horizon)
        y_horizons.append(y_growth)
    return y_horizons

def cum_gr(series, horizon ,yearfreq=4): 
    ## Compute the compound annualized quarterly growth rate over a certain horizon
    cagr = ((series.shift(-horizon)/series)**(1/horizon))-1
    ## Need to annualize it now
    annual_cagr = ((1+cagr)**yearfreq) -1
    return(100*annual_cagr)

def yoy_gr(series, horizon, yearfreq=4): 
    ## We assume that the growth rate is quarterly. In the future, rather than having +4, should use an index period
    yoy_gr = (series.shift(-horizon)/series.shift(-horizon+yearfreq))-1
    return(100*yoy_gr)

def gen_scenario_data(shockvar_dict, partition_groups, original_data, retropolate='Yes'):
    scenarios = {}
    for scenario, shockvars in shockvar_dict.items():
        df_shockedvar, df_shockedgrp = gen_shocked_PCA(shockvars, partition_groups, original_data, retropolate)
        scenarios[scenario] = df_shockedgrp

    baseline_data, partition_load = PCA_partitioning(original_data, partition_groups, retropolate)
    scenarios['baseline'] = baseline_data
    return scenarios

def gen_shocked_PCA(shockvars, partition_groups, original_data, retropolate='Yes'):
    # perform PCA
    df_partition, partition_load = PCA_partitioning(original_data, partition_groups, retropolate)

    # generate some shocks
    df_shockedvar, df_shockedgrp = gen_relation(shockvars, partition_groups, original_data, df_partition)

    return df_shockedvar, df_shockedgrp

# Calculate shock relations
def gen_relation(shockdict, partition_groups, original_data, df_partition):
    df_shockedvar = pd.DataFrame(index=original_data.index)
    df_shockedgrp = df_partition.copy()
    for var, shock in shockdict.items():
        if shock['shocktype']=='By +/- STD':
            if var in partition_groups.keys():
                std = np.nanstd(df_shockedgrp[var])
            else:
                df_shockedvar[var] = original_data[var]
                std = np.nanstd(original_data[var].values)
                df_shockedvar[var+'_shocked']=df_shockedvar[var]+std*shock['shockvalue']
        elif shock['shocktype']=='By +/- percentage' and (var not in partition_groups.keys()):
            df_shockedvar[var]=original_data[var]
            df_shockedvar[var+'_shocked']=df_shockedvar[var]*(1+shock['shockvalue'])
        for group, compvars in partition_groups.items():  
            if var in compvars:
                df_var = original_data[['date',var]].dropna()
                df_part = df_partition[['date',group]].dropna()

                sdate=max(min(df_var['date'].values),(min(df_part['date'].values)))
                edate=min(max(df_var['date'].values),(max(df_part['date'].values)))

                df_var=df_var[(df_var['date']>=sdate) & (df_var['date']<=edate)]
                df_part=df_part[(df_part['date']>=sdate) & (df_part['date']<=edate)]
                
                # use correlation to understand how much of the shock will leak into the group 
                cov=np.corrcoef(df_var[var].values,df_part[group].values)[0][1]
                if shock['shocktype']=='By +/- STD':
                    # !!! Will have to review the logic behind this !!!
                    df_shockedgrp[group]= df_shockedgrp[group]+std*shock['shockvalue']*cov
                elif shock['shocktype']=='By +/- percentage':
                    df_shockedgrp[group]= df_shockedgrp[group]+original_data[var]*shock['shockvalue']*cov               
            elif var==group:
                if shock['shocktype']=='By +/- STD':
                    df_shockedgrp[group]= df_shockedgrp[group]+std*shock['shockvalue']
                elif shock['shocktype']=='By +/- percentage':                
                    df_shockedgrp[group]= df_shockedgrp[group]+df_shockedgrp[group]*shock['shockvalue']
    
    # set index for shockvar
    df_shockedvar.index = original_data['date']
    return df_shockedvar, df_shockedgrp

def PCA_partitioning(data, dict_groups, retropolate='Yes'):
    # partition input data
    first_valid = data.apply(lambda x: x.first_valid_index())
    last_valid = data.apply(lambda x: x.last_valid_index())
    start_index = max([min([first_valid[value] for value in values]) for key, values in dict_groups.items()])
    end_index = min([min([last_valid[value] for value in values]) for key, values in dict_groups.items()])

    dict_input_partition = {
        'sdate': data['date'][start_index].to_pydatetime(),
        'edate': data['date'][end_index].to_pydatetime(),
        'retropolate': retropolate,
        }
    return run_partition(dict_input_partition, dict_groups, data)

def run_partition(dict_input_partition, dict_groups, df_partition):
    '''
    Main run function for step 1, partition.

    Takes in as arguments a dict for input parameters
    and a df for data. Outputs a dict for output parameters.

    Does partitioning and returns a dict of output parameters.
    ** This function should be independent of any Excel input/output
    and be executable as a regular Python function independent of Excel. **
    '''
    # ------------------------
    # Get parameters from
    # dict_input_partition
    # ------------------------
    sdate   = dict_input_partition['sdate']
    edate   = dict_input_partition['edate']
    df_partition  = df_partition.set_index(df_partition['date'], drop=False)
    
    # ------------------------
    # Run the partition
    # ------------------------
    retroframe, retroload = partition_retro(df_partition,dict_groups, sdate, edate)
    retroframe.reset_index(drop=True, inplace=True)
    return retroframe, retroload

#Function to generate retropolated partition in a time period
def partition_retro(dall, groups_dict, sdate, edate):
    # Some data treatment
    dall = dall.fillna(method='ffill').copy()
    dall = dall[(dall['date']>=sdate) & (dall['date']<=edate)]

    # Generating all cutoffs in the period, sorted from latest to earliest
    [cutoffs, complete_group] = gen_cutoff(dall=dall, groups_dict=groups_dict,startdate=sdate, enddate=edate)

    if (cutoffs==-1):
        return dall.head(), dall.head()

    if len(cutoffs)==0:
        return dall.head(), dall.head()

    # Generating the parition for the latest cutoff            
    [dp1, dl]=p_cutoff(dall, groups_dict, cutoffs[0])
    dpo=dp1
    for i in range(1,len(cutoffs)):
        [dpn,dln]=p_cutoff(dall,groups_dict,cutoffs[i])
        dpr=retropolate(dfearly=dpn, dflate=dpo, complete_early=complete_group[i], groups_dict=groups_dict)
        dpo=dpr.copy()
        retrovar=" "
        for e in groups_dict:
            if e not in complete_group[i]:
                retrovar+=e+", "    
    dl['cutoff']=sdate
    dl=dl[['variable','cutoff','loadings','group','variance_ratio']]
        
    # Compute the zscore for the final frame to makes them consistent
    group_vars = [x for x in groups_dict.keys()]
    for group in group_vars:
        dpo[group] = zscore(dpo[group])

    dretro_final = dpo
    dretro_final.index=dretro_final['date']
    dretro_final.index.name=None

    if 'country' in dretro_final.columns:
        dretro_final.drop(columns='country', inplace=True)
    return dretro_final, dl

# Given two frame of signle country retroplate late frame to early frame, 
# return the retroplated frame
def retropolate(dfearly,dflate,complete_early,groups_dict):
###############################################################################

    ###########################
    ###TODO Remove country#####
    dfearly['country']=0
    dflate['country']=0
    dfearly.index.name=None
    dflate.index.name=None
    ###########################
    #dload = pd.read_excel(gv.final_data_dir + '/Partitions_late.xlsx',
    #                    sheetname='Loadings') 

    ## Select the data of interest
    ## This part can be removed as it shoud be done outside of the function.
    group_vars = [x for x in groups_dict.keys()]
    all_vars = ['country', 'date'] + group_vars

    de = dfearly.loc[:,all_vars].copy()
    dl = dflate.loc[:,all_vars].copy()
## Sort the frames
    de = dfearly.sort_values(by=['country', 'date'], ascending=[1,1])
    dl = dflate.sort_values(by=['country', 'date'], ascending=[1,1])


# For every country, compute the reverse growth rate based on early data
###############################################################################
## Compute the reverse delta (from future to now, data inverted)
    for pvar in group_vars:
        rgr_n = '{}_rgr'.format(pvar)

    ## Need to normalize: compute the zscore, per country 
        de[pvar] = de.groupby(['country'])[pvar].apply(zscore)   
        dl[pvar] = dl.groupby(['country'])[pvar].apply(zscore)
    
    ## Compute the delta, per country (pay attention to the order, future second)
        de[rgr_n] = de.groupby(['country'])[pvar].apply(lambda x: x - x.shift(-1))
        dl[rgr_n] = dl.groupby(['country'])[pvar].apply(lambda x: x - x.shift(-1))     

###############################################################################
# Index creation using the reverse delta
## Dulani's trick: sum for small numbers, growth rate for large number !!
###############################################################################

    # 1. Identify the missing dates from the late frame
    # dec = de.loc[de.country==pays,:]
    # dmc = dm.loc[dm.country==pays,:]
    # dlc = dl.loc[dl.country==pays,:]

    ####### From late to middle
    late_missing_dates = sorted(list(set(de.date) - set(dl.date)))
    late_start_date = min(dl.date)
    

    ## Isolate the middle frame without long time frame
    ef = de.loc[de.date < late_start_date, :]
    ef = ef.sort_values(by='date', ascending=0) # Reverse cum sum !!


    # 2. Compute the cumulative growth rate based only on the recent frame
    for pv in group_vars:
        ef['{}_cum_rgr'.format(pv)] = ef['{}_rgr'.format(pv)].cumsum().copy()

    ## 3. Using cumulative sum, create the missing frame
    mgr_frames_list = list()

    ## Retroplating for every group, only incomplete group will be updated.
    for group in group_vars:
        
        start_val = dl.loc[dl['date'] == min(dl['date']),group].values[0]
        dng = pd.DataFrame(index=late_missing_dates, columns=['date'])
        dng['date'] = dng.index.values
        dng.index.name=None
        dng['country'] = de['country'].values[0]
        dng = dng.sort_values(by='date', ascending=0)

        
        gr_cum = '{}_cum_rgr'.format(group)
        dng_f = dng.merge(ef[['date', gr_cum]], on=['date'], how='left')
        #dng_f[group] = dng_f[gr_cum] + start_val # Increment the value

        #If group in the early frame is complete, no retroplation is needed.
        #Use the value in early group

        if group in complete_early:
            dng_f[group]=ef[group].values
        else:
            dng_f[group] = dng_f[gr_cum] + start_val


        dng_f.index=dng_f['date'].values
        dng_f.index.name=None
        mgr_frames_list.append(dng_f)
        
      
    ## Merge the new groups into a early augmented frame
    dea = mgr_frames_list[0]
    for frame in mgr_frames_list[1:]:
        dea = pd.merge(dea, frame, on=['date', 'country'])
    dea.index=dea['date'].values
    dea.index.name=None

    ## Merge late, early augmented

    d_complete = pd.concat([dl[all_vars], dea[all_vars]],axis='index')
    

    d_complete=d_complete.sort_values(by=['country', 'date'])
    dfearly=dfearly.sort_values(by=['country', 'date'])
    
    ## complete group fix
    for group in complete_early:
        d_complete[group]=dfearly[group]

    return d_complete

# Function to generate partition cutoff points and completed groups
# at the coressponding cutoff ponit. Completed group will not be retropolated.
def gen_cutoff (dall="default", groups_dict={}, startdate=date(year=1,month=1,day=1), enddate=date(year=9999,month=12,day=31)):
    if len(dall)==0:
        print("No data found")
        return -1
    
    dall = dall.fillna(method='ffill').copy()
    dall = dall[dall.date>=startdate]
    dall = dall[dall.date<=enddate]
    partition_dict = groups_dict # Variables per group (price, leverage, etc.)
    
    set_date=set([])
    for key, values in partition_dict.items():
        for v in values:
            t=dall[v].first_valid_index() 
            if t not in set_date:
                set_date.add(t)
    dates=list(set_date)
    dates.sort(reverse=True)
    complete_groups=[]
    for d in dates:                
        tmp_c_key=[]
        for key, values in partition_dict.items():
            
            complete_key=True
            empty_key=True
            for v in values:
                if dall[v].first_valid_index()>d:
                    complete_key=False
                else:
                    empty_key=False
            if empty_key:
                for v in values:
                    print(v,dall[v].first_valid_index())
                print("In the given time period some groups are complete empty. No feasible partition can be made")
                return -1,-1
            else:
                if complete_key:
                    tmp_c_key.append(key)
        complete_groups.append(tmp_c_key)
            
            
    return dates,complete_groups

def p_cutoff(dall,groups_dict,cutoff):
    df = dall.loc[dall.date >= cutoff].copy()
    partition_dict = groups_dict
    variables=[]
    label_dict={}
    for key, values in partition_dict.items():
        variables.extend(values)
        for e in values:
            label_dict[e]=e
    
    c_id_dict = {'cutoff' : cutoff.strftime("%Y-%m-%d"), 
                 'variables': repr(variables)}
    
    p = Partition(df, partition_dict, reduction='PCA')
    dp = p.partition # Run the partition on the full frame

    dp = add_id(dp, c_id_dict)
    dp.loc[:,'date'] = dp.index
    dp.index.name=None
    ## Loading from the partitioning
    dl = p.loading; dl = add_id(dl, c_id_dict)
    dl.insert(0, 'variable_o', dl.index)
    dl.loc[:,'variable'] = dl.variable_o.apply(lambda x : label_dict[x])
    return [dp,dl]

# Data partitioning
class Partition(object):
    ## Initializer
    def __init__(self, data, groups_dict, reduction='PCA'):

        ## Parameters
        self.reduction = reduction

        ## Clean the dataset according to the type of reduction
        self.data = data.dropna(axis=0, how='all').dropna(axis=1,how='any').fillna(method='ffill').copy()

        ## Remove constant columns (create problem in the partitioning)
        self.data = self.data.loc[:, self.data.apply(pd.Series.nunique) != 1]
        
        ## Populate the groups only with the variables available in the frame
        self.var_dict = {k:[x for x in groups_dict[k] if x in self.data.columns]
                         for k in groups_dict.keys()}

        self.partition_fit_group, self.loading = self.__partition_fit_PCA()

        for group in sorted(list(self.partition_fit_group.keys())):
            setattr(self.partition_fit_group[group], 'fit',
                    self.partition_fit_group[group].fit_transform)
            
        # By default, using the original data (can be customized)
        self.partition = zscore(self.partition_data(self.data)) 

    ## Methods
    def __partition_fit_PCA(self):
        """ Run the data partitioning using Principal Component Analysis """
        groups = sorted(list(self.var_dict.keys()))
        pca_fit_group = dict()
        loadings_frame = list()
        
        for group in groups:
            var_list = self.var_dict[group]
            if len(var_list) > 1: # Run the partition
                # Partitionning
                dg = self.data.loc[:, var_list].copy()
                X = scale(dg) # Need to scale the variables before partitioning

                ## Fit the PCA
                pca_fit = PCA(n_components=1).fit(X)
                pca_fit_group[group] = pca_fit

                ## Store the loadings
                dl = pd.DataFrame(pca_fit.components_, index=['loadings'],
                                  columns=var_list).transpose()
                dl['group'] = group
                dl['variance_ratio']=pca_fit.explained_variance_ratio_[0]
                dl['variable'] = dl.index
                loadings_frame.append(dl)
                
            elif len(var_list) == 1: # Loadings are 1
                dl = pd.DataFrame(index=var_list)
                dl['loadings'] = 1
                dl['variance_ratio']=1
                dl['group'] = group
                dl['variable'] = var_list[0]
                loadings_frame.append(dl)

            else: # Empty group: no loading
                dl = pd.DataFrame(columns=['loadings', 'group', 'variable'])
                dl['loadings'] = np.nan
                dl['variance_ratio']=np.nan
                dl['group'] = group
                dl['variable'] = np.nan
                loadings_frame.append(dl)

        dloading = pd.concat(loadings_frame)

        # Return the fit method and the associated loadings                
        return((pca_fit_group, dloading))                        

    def partition_data(self, dataframe):
        """ Return the aggregated data """
        # From the previous step, extract the fitting for each group
        groups = sorted(list(self.var_dict.keys()))
        pfit = self.partition_fit_group

        ## Prepare to store the data and the loadings
        da = pd.DataFrame(index=dataframe.index)
        
        for group in groups:
            var_list = self.var_dict[group]
            if len(var_list) > 1: # Use the loadings from the partition fit
                dg = dataframe.loc[:, var_list].copy()
                
                # Scale the variables
                X = scale(dg) 
                              
                ## Generate the data using the partitioning fit
                da[group] = pfit[group].transform(X)
                                    
                
            elif len(var_list) == 1: # Simply keep the variable as it is
                da[group] = dataframe.loc[:, var_list[0]]
        
            else: # Empty group
                da[group] = np.nan

        return(da)

# Fuction to do partion for one cutoff time. Return partion and loading
def add_id(df, id_dict):
    """ Add identifiers variables to a pandas frame """
    variables_id = sorted(list(id_dict.keys()))
    for v, var in enumerate(variables_id):
        df.insert(v, var, id_dict[var])
    return(df)

# Zscore correction
def zscore(series):
    return((series - series.mean())/series.std(ddof=0))

### Dist fit functions ###
# Ancillary functions, cf. Zhu and Galbraith JoE 2010
def K_plain(nu):
    top = gamma((nu+1)/2)
    bottom = np.sqrt(pi*nu)*gamma(nu/2)
    return(top/bottom)

def alpha_star_plain(alpha, nu1, nu2):
    top = alpha*K(nu1)
    bottom = alpha*K(nu1) + (1-alpha)*K(nu2)
    return(top/bottom)

K = np.vectorize(K_plain, otypes=[np.float], cache=False)
alpha_star = np.vectorize(alpha_star_plain, otypes=[np.float], cache=False)

# T-Skew distance between a set of estimated quantiles and theoretical ones
def quantile_interpolation(alpha, cond_quant_dict):
    """ 
    Quantile interpolation function, following Schmidt and Zhu (2016) p12
    - Alpha is the quantile that needs to be interpolated
    - cond_quant_dict is the dictionary of quantiles to interpolate on 

    Return:
    - The interpolated quantile
    """

    ## List of quantiles
    qlist = sorted(list(cond_quant_dict.keys()))
    min_q = min(qlist)
    max_q = max(qlist)

    ## Fix the base quantile function (usually a N(0,1))
    base = norm.ppf

    ## Considering multiple cases
    if alpha in qlist: ## No need to interpolate, just on the spot !!
        interp = cond_quant_dict[alpha]

    elif alpha < min_q: ## The left edge
        ## Compute the slope (page 13) 
        b1_up = (cond_quant_dict[max_q] - cond_quant_dict[min_q])
        b1_low = base(max_q) - base(min_q)
        b1 = b1_up/b1_low

        ## Compute the intercept (page 12)
        a1 = cond_quant_dict[min_q] - b1*base(min_q)

        ## Compute the interpolated value
        interp = a1 + b1*base(alpha)
        
    elif alpha > max_q: # The right edge (same formula)
        ## Compute the slope (page 13) 
        b1_up = (cond_quant_dict[max_q] - cond_quant_dict[min_q])
        b1_low = base(max_q) - base(min_q)
        b1 = b1_up/b1_low

        ## Compute the intercept (page 12)
        a1 = cond_quant_dict[min_q] - b1*base(min_q)

        ## Compute the interpolated value
        interp = a1 + b1*base(alpha)

    else: # In the belly
        ## Need to identify the closest quantiles
        local_min_list = [x for x in qlist if x < alpha]
        local_min = max(local_min_list) # The one immediately below

        local_max_list = [x for x in qlist if x > alpha]
        local_max = min(local_max_list) # The one immediately above

        # Compute the slope
        b_up = (cond_quant_dict[local_max] - cond_quant_dict[local_min])
        b_low = base(local_max) - base(local_min)
        b = b_up/b_low

        # Compute the intercept
        a = cond_quant_dict[local_max] - b*base(local_max)
        
        ## Compute the interpolated value
        interp = a + b*base(alpha)

    ## Return the interpolated quantile    
    return(interp)

# Uncrossing
def quantile_uncrossing(cond_quant_dict, method='linear'):
    """ 
    Uncross a set of conditional_quantiles using Cherzonukov et al 2010
    Via bootstrapped rearrangement
    
    Input:
    - A dictionary of quantile: conditional quantiles
    - Interpolation method: either linear or probabilistic. 
    The probabilistic quantile interpolation follows Zhu and Schmidt 2016

    Output:
    - A dictionary of quantile: uncrossed conditional quantiles

    """

    ## List of quantiles
    ql = sorted(list(cond_quant_dict.keys()))
    cond_quant = [cond_quant_dict[q] for q in ql] # Because dict is not ordered
    np.random.seed(2018)
    ## Check if the quantiles are crossing in the first place
    if sorted(cond_quant) == cond_quant:
        cond_quant_uncrossed_dict = cond_quant_dict
    else:
        if method=='linear':         
            ## Use a linear interpolation for the quantile function
            inter_lin = interpolate.interp1d(ql, cond_quant,
                                             fill_value='extrapolate')

            ## Bootstrap the quantile function
            bootstrap_qf = inter_lin(np.random.uniform(0,1,1000))

            ## Now compute the percentiles of the bootstrapped quantiles 
            cond_quant_uncrossed = [np.percentile(bootstrap_qf, 100*q)
                                    for q in ql]

            ## They are the uncrossed quantiles !
            cond_quant_uncrossed_dict = dict(zip(ql, cond_quant_uncrossed))

        elif method=='probabilistic':
            ## Use Schmidt and Zhu (2016) approach
            bootstrap_qf = [quantile_interpolation(u, cond_quant_dict)
                            for u in np.random.uniform(0,1,1000)]

            ## Now compute the percentiles of the bootstrapped quantiles 
            cond_quant_uncrossed = [np.percentile(bootstrap_qf, 100*q)
                                    for q in ql]

            ## They are the uncrossed quantiles !
            cond_quant_uncrossed_dict = dict(zip(ql, cond_quant_uncrossed))

        else:
            raise ValueError('Interpolation method misspecified')
            
    ## Return the uncrossed quantiles    
    return(cond_quant_uncrossed_dict)

def tskew_distance(quantile_list, cond_quant, 
                   df, loc, scale, skew):
    """ Return the distance between theoretical and actual quantiles"""

    def tskew_tau(tau):
        """ Function which only depends on a given tau """
        return(tskew_ppf(tau, df=df, loc=loc, scale=scale, skew=skew))

    tskew_ppf_vectorized = np.vectorize(tskew_tau, otypes=[np.float])
    
    theoretical_quant = tskew_ppf_vectorized(quantile_list)  

    diff = np.subtract(theoretical_quant, cond_quant)
    diff2 = np.power(diff,2)    
    msse = np.sum(diff2)
    
    loc_tskew=loc
    for i in range(len(quantile_list)):
        if quantile_list[i]==0.25:
            lowq=cond_quant[i]
        if quantile_list[i]==0.75:
            highq=cond_quant[i]
    alpha=10
    if loc_tskew<=highq and loc_tskew>=lowq:
        penalty=0
    else:
        penalty=alpha*min((lowq-loc_tskew)**2,(highq-loc_tskew)**2)
    mssepen=msse+penalty
    return(mssepen)

# Optimal TSkew fit based on a set of conditional quantiles and a location
def tskew_fit(conditional_quantiles, fitparams):
    """ 
    Optimal TSkew fit based on a set of conditional quantiles and a location
    Inputs:
        - conditional_quantiles (dictionary): quantiles & conditional value
        - loc: location. Can be estimated as a conditional mean via OLS

    Output:
        - A dictionary with optimal scale and skewness, as well as df and loc 
    """
    conditional_quantiles=quantile_uncrossing(conditional_quantiles)
    quantile_list = np.sort(list(conditional_quantiles.keys()))
    
    ######################
    #Generate Parameters##
    ######################
    if fitparams['skew_low']!='Free':
        pass
    

    ## Interquartile range (proxy for volatility)
    try:
        IQR = np.absolute(conditional_quantiles[0.75] - conditional_quantiles[0.25])
        IQR = np.clip(IQR, 1, 10) # Avoid degenerate interquartile range
        # At least 1 pp growth and at most 10 ppt growth in the interquartile
    except:
        raise ValueError('Need to provide estimate for 25% and 75% quantiles')

    # Avoid crossing: sort
    cond_quant = np.sort(list(conditional_quantiles.values()))
    
    if fitparams['var_low']['constraint']=='Fixed':
        scale_down=fitparams['var_low']['value']
    else:
        scale_down = np.sqrt(IQR)/2 +0.1# Good lower bound approximation
        
    if fitparams['var_high']['constraint']=='Fixed':
        scale_up=fitparams['var_high']['value']            
    else:
        scale_up = IQR/1.63 + 0.2# When skew=1, variance exactly = IQR/1.63 
        
    if fitparams['skew_low']['constraint']=='Fixed':
        skew_low =fitparams['skew_low']['value']
    else:
        skew_low = 0.1 # Default lower bound approximation
        
    if fitparams['skew_high']['constraint']=='Fixed':
        skew_high=fitparams['skew_high']['value']
        x0_f = [(scale_down+scale_up)/2, (skew_low + skew_high)/2] # Initial values            
    else:
        skew_high = 3 # Default higher bound approximation
        x0_f = [IQR/1.63 + 0.1, 1]
    
    if fitparams['mode']['constraint']=='Fixed':
        loc=fitparams['mode']['value']
    else:
        loc=conditional_quantiles[0.5]
    
    if fitparams['dof']['constraint']=='Fixed':        
        o_df = fitparams['dof']['value'] # Degrees of freedom
    else:
        o_df = 2


    ## Two values optimizer: on both conditional variance and skewness
    def mult_obj_distance(x): # x is a vector
        """ Multiple parameters estimation """
        ## Unpack the vector
        scale = x[0]
        skew = x[1]

        # Run the optimizer
        obj = tskew_distance(quantile_list=quantile_list,
                             cond_quant=cond_quant,
                             df=o_df, loc=cond_mean, scale=scale, skew=skew)
        return(obj)
    
    ## Two values optimizer: on both conditional variance and skewness
    def mult_obj_distance3(x): # x is a vector
        """ Multiple parameters estimation """
        ## Unpack the vector
        scale = x[0]
        skew = x[1]
        dloc = x[2]
        # Run the optimizer
        obj = tskew_distance(quantile_list=quantile_list,
                             cond_quant=cond_quant,
                             df=o_df, loc=dloc, scale=scale, skew=skew)
        return(obj)
    
    
    ## Run the optimizer
    locs = loc+0.5
    cond_mean=0
    cdmeanmax=loc+10
    cdmeanmin=loc-10
    
    if fitparams['mode']['constraint']!='Free':
    #bisection optimize for location
        maxit=0
        while maxit<100 and abs(locs-loc)>0.00001:
            
            cond_mean=(cdmeanmin+cdmeanmax)/2
            
    # Fix the boundaries to avoid degenerative distributions
            bnds_f = ((scale_down, scale_up), (skew_low , skew_high))
            res = minimize(mult_obj_distance, x0=x0_f,
                           bounds=bnds_f, method='SLSQP',
                           options={'maxiter':1000,  'ftol': 1e-04, 'eps': 1.5e-06})
        
            o_scale, o_skew  = res.x
            locs=cond_mean
            if locs>loc:
                cdmeanmax=cond_mean
            else:
                cdmeanmin=cond_mean
            maxit+=1
        
    ## Package the results into a dictionary
        fit_dict = {'loc': float("{:.4f}".format(cond_mean)),
                    'df': int(o_df),
                    'scale': float("{:.4f}".format(o_scale)),
                    'skew': float("{:.4f}".format(o_skew))}
    
        return(fit_dict)
    else:
        x0_f.append(0) # Initial values
    
        # Fix the boundaries to avoid degenerative distributions
        bnds_f = ((scale_down, scale_up),  (skew_low , skew_high), (-20,20))
        res = minimize(mult_obj_distance3, x0=x0_f,
                       bounds=bnds_f, method='SLSQP',
                       options={'maxiter':1000,  'ftol': 1e-04, 'eps': 1.5e-06})
        
        o_scale, o_skew, o_loc  = res.x
        
    ## Package the results into a dictionary
        fit_dict = {'loc': float("{:.4f}".format(o_loc)),
                    'df': int(o_df),
                    'scale': float("{:.4f}".format(o_scale)),
                    'skew': float("{:.4f}".format(o_skew))}
        return(fit_dict)

def asymt_distance(quantile_list, cond_quant, 
                   skew, kleft, kright, loc, scale,ols):
    """ Return the distance between theoretical and actual quantiles"""

    def asymt_tau(tau):
        """ Function which only depends on a given tau """
        return(asymt_ppf(tau, alpha=skew, nu1=kleft, nu2=kright, mu=loc, sigma=scale))
        #asymt_ppf(p, alpha=0.5, nu1=1, nu2=1, mu=0, sigma=1)
    asymt_ppf_vectorized = np.vectorize(asymt_tau, otypes=[np.float])
    
    theoretical_quant = asymt_ppf_vectorized(quantile_list)  

    diff = np.subtract(theoretical_quant, cond_quant)
    diff2 = np.power(diff,2)    
    msse = np.sum(diff2)

    loc_tskew=loc
    for i in range(len(quantile_list)):
        if quantile_list[i]==0.25:
            lowq=cond_quant[i]
        if quantile_list[i]==0.75:
            highq=cond_quant[i]
    
    a1=10
    if loc_tskew<=highq and loc_tskew>=lowq:
        penalty=0
    else:
        penalty=a1*min((lowq-loc_tskew)**2,(highq-loc_tskew)**2)
    olsdiff=2*(ols[0]-asymt_mean(alpha=skew, nu1=kleft, nu2=kright, mu=loc, sigma=scale))**2

    mssepen=msse+penalty+olsdiff
    return(mssepen)

# Optimal TSkew fit based on a set of conditional quantiles and a location
def asymt_fit(conditional_quantiles, fitparams, olsmean):
    """ 
    Optimal TSkew fit based on a set of conditional quantiles and a location
    Inputs:
        - conditional_quantiles (dictionary): quantiles & conditional value
        - loc: location. Can be estimated as a conditional mean via OLS

    Output:
        - A dictionary with optimal scale and skewness, as well as df and loc 
    """
    conditional_quantiles=quantile_uncrossing(conditional_quantiles)
    quantile_list = np.sort(list(conditional_quantiles.keys()))
    ols=[olsmean]
    ######################
    #Generate Parameters##
    ######################
    if fitparams['skew_low']!='Free':
        pass
    

    ## Interquartile range (proxy for volatility)
    try:
        IQR = np.absolute(conditional_quantiles[0.75] - conditional_quantiles[0.25])
        IQR = np.clip(IQR, 1, 10) # Avoid degenerate interquartile range
        # At least 1 pp growth and at most 10 ppt growth in the interquartile
    except:
        raise ValueError('Need to provide estimate for 25% and 75% quantiles')


    ## Upper-bound for the scale
#    scale_up = IQR/1.63 + 0.2# When skew=1, variance exactly = IQR/1.63 
#    scale_down = np.sqrt(IQR)/2 +0.1# Good lower bound approximation
    
    # Avoid crossing: sort
    cond_quant = np.sort(list(conditional_quantiles.values()))
    
    ## Define the boundaries of the conditional quantiles
    #min_o = np.nanmin(cond_quant)
    #max_o = np.nanmax(cond_quant)

    ## Conditional mean can not be inside the conditional quantiles
    ## Else the distribution would be completely degenerated

    # cond_mean = np.clip(loc, min_o, max_o)
    x0_f = [1, 0.5, 1,1]
    
    if fitparams['var_low']['constraint']=='Fixed':
        scale_down=fitparams['var_low']['value']
    else:
        scale_down = 0.01# Good lower bound approximation
        
    if fitparams['var_high']['constraint']=='Fixed':
        scale_up=fitparams['var_high']['value']            
    else:
        scale_up = 5# When skew=1, variance exactly = IQR/1.63 
        
    if fitparams['skew_low']['constraint']=='Fixed':
        skew_low =fitparams['skew_low']['value']
    else:
        skew_low = 0.01 # Default lower bound approximation
        
    if fitparams['skew_high']['constraint']=='Fixed':
        skew_high=fitparams['skew_high']['value']
        x0_f = [(scale_down+scale_up)/2, (skew_low + skew_high)/2,1,1] # Initial values            
    else:
        skew_high = 0.99 # Default higher bound approximation
        x0_f = [1, 0.5, 1,1]
    
    if fitparams['mode']['constraint']=='Fixed':
        loc=fitparams['mode']['value']
    else:
        loc=conditional_quantiles[0.5]
    

    if olsmean<conditional_quantiles[0.5]:
        skew_low=0.5+2*abs(conditional_quantiles[0.5]-olsmean)/abs(conditional_quantiles[0.5])
        skew_high=max(skew_low,skew_high)
        x0_f=[1,max(skew_low,0.8),2,2]
    else:
        skew_high=0.5-2*abs(conditional_quantiles[0.5]-olsmean)/abs(conditional_quantiles[0.5])
        skew_low=min(skew_low,skew_high)
        x0_f=[1,min(skew_high,0.2),2,2]
    ## Two values optimizer: on both conditional variance and skewness
    def mult_obj_distance(x): # x is a vector
        """ Multiple parameters estimation """
        ## Unpack the vector
        scale = x[0]
        skew = x[1]
        dkleft=x[2]
        dkright=x[3]
        # Run the optimizer
        obj = asymt_distance(quantile_list=quantile_list,
                             cond_quant=cond_quant,
                             kleft=dkleft,kright=dkright, loc=cond_mean, scale=scale, skew=skew , ols=ols)
        
        return(obj)
    
    
    
    ## Two values optimizer: on both conditional variance and skewness
    def mult_obj_distance3(x): # x is a vector
        """ Multiple parameters estimation """
        ## Unpack the vector
        scale = x[0]
        skew = x[1]
        dkleft=x[2]
        dkright=x[3]
        dloc = x[4]
        # Run the optimizer
        obj = asymt_distance(quantile_list=quantile_list,
                             cond_quant=cond_quant,
                             kleft=dkleft,kright=dkright, loc=dloc, scale=scale, skew=skew, ols=ols)
        return(obj)
    
    
    ## Run the optimizer
    cond_mean=loc
    
    if fitparams['mode']['constraint']!='Free':
    #bisection optimize for location
        bnds_f = ((scale_down, scale_up),  (skew_low , skew_high),(0.2,10),(0.2,10))
        res = minimize(mult_obj_distance, x0=x0_f,
                       bounds=bnds_f, method='SLSQP',
                       options={'maxiter':1000,  'ftol': 1e-06, 'eps': 1.5e-08})
        
        o_scale, o_skew , o_kleft, o_kright = res.x
        fit_dict = {'loc': float("{:.4f}".format(loc)),
                    'kleft':float("{:.4f}".format(o_kleft)),
                    'kright':float("{:.4f}".format(o_kright)),
                    'scale': float("{:.4f}".format(o_scale)),
                    'skew': float("{:.4f}".format(o_skew))}
    
        return(fit_dict)
    else:
        x0_f.append(conditional_quantiles[0.25]) # Initial values
        # Fix the boundaries to avoid degenerative distributions
        bnds_f = ((scale_down, scale_up),  (skew_low , skew_high), (0.2,1000),(0.2,1000),(-20,20))
        res = minimize(mult_obj_distance3, x0=x0_f,
                       bounds=bnds_f, method='SLSQP',
                       options={'maxiter':1000,  'ftol': 1e-06, 'eps': 1.5e-08})
        
        o_scale, o_skew, o_kleft, o_kright ,o_loc  = res.x
        
    ## Package the results into a dictionary
        fit_dict = {'loc': float("{:.4f}".format(o_loc)),
                    'kleft':float("{:.4f}".format(o_kleft)),
                    'kright':float("{:.4f}".format(o_kright)),
                    'scale': float("{:.4f}".format(o_scale)),
                    'skew': float("{:.4f}".format(o_skew))}
        return(fit_dict)

# Percentage point function (quantile function) of a TSkew distribution
def tskew_ppf(tau, df, loc, scale, skew):
    """
    Quantile function of the tskew distribution 
    Based on the formula in Giot and Laurent (JAE 2003 pp. 650)
    - tau = the quantile
    - df: degrees of freedom (>1)
    - location: mean of the distribution
    - scale: standard deviation of the distribution (>0)
    - skew: skewness parameter (>0, if ==1: no skew, <1: left skew, >1 right)
    
    NB: I had to parametrize the formula differently (was wrong in their paper)
    """

    threshold = 1/(1+np.power(skew,2))
    if tau < threshold:
        adj_tau = (tau/2)*(1+np.power(skew,2))
        non_stand_quantile = (1/skew)*t.ppf(adj_tau, df=df, loc=0, scale=1)
    elif tau >= threshold:
        adj_tau = ((1-tau)/2)*(1+(1/np.power(skew,2)))
        non_stand_quantile = -skew*t.ppf(adj_tau, df=df, loc=0, scale=1)
    else:
        raise ValueError('Parameters misspecified')
    
    quantile = loc + (non_stand_quantile*scale) # Pay attention to this one !
    
    return(quantile)

#Probability density function of a TSkew distribution
def tskew_pdf(x, df, loc, scale, skew):    
    """
    Density function of the tskew distribution 
    Based on the formula in Giot and Laurent (JAE 2003 pp. 650)
    - x = the value to evaluate
    - df: degrees of freedom (>1)
    - location: mean of the distribution
    - scale: standard deviation of the distribution
    - skew: skewness parameter (>0, if ==1: no skew, <1: left skew, >1 right)
    
    NB: I had to parametrize the formula differently to get consistent results
    
    """
    cons = (2/(skew + (1/skew)))/scale
    norm_x =  (x-loc)/scale
    if x < loc :
        pdf = cons*t.pdf(skew*norm_x, df, loc=0, scale=1) # Symmetric t pdf
    elif x >= loc:
        pdf = cons*t.pdf(norm_x/skew, df, loc=0, scale=1) # Symmetric t pdf
    else:
        raise ValueError('Incorrect parameters')

    return(pdf)

# Cumulative distribution of a TSkew distribution:
def tskew_cdf(x, df, loc, scale, skew):    
    """
    Density function of the tskew distribution 
    Based on the formula in Giot and Laurent (JAE 2003 pp. 650) 
    and Lambert and Laurent (2002) pp. 10
    - x = real value on the support to evaluate
    - df: degrees of freedom (>1)
    - location: mean of the distribution
    - skew: skewness parameter (>0, if ==1: no skew, <1: left skew, >1 right)
    
    NB: I had to parametrize it differently in order to get consistent results
    
    """
    
    sk2 = np.power(skew, 2); inv_sk2 = 1/sk2
    norm_x1 = (x-loc)/scale
    #norm_x2 = x-(loc/scale)
    if x < loc:
        # t.cdf() is the symmetric t cdf
        cdf = (2/(1+sk2))*t.cdf(skew*norm_x1, df, loc=0, scale=1) 
    elif x >= loc:
        cdf = 1 - (2/(1+inv_sk2))*t.cdf(-norm_x1/skew, df, loc=0, scale=1)
    else:
        raise ValueError('Incorrect parameters')

    return(cdf)

# PPF of the Assymetric student T, cf. Zhu and Galbraith JoE 2010
def asymt_ppf(p, alpha=0.5, nu1=1, nu2=1, mu=0, sigma=1):

    """ 
    Following Zhu and Galbraith, pp. 299-300  
    Alpha is the skewness, nu1 and nu2 are the left and right kurtosis
    mu is location (mode) and sigma the scale (variance)
    """
       
    left_bracket = np.min([p, alpha])/(2*alpha)
    left_block = 2*alpha_star(alpha, nu1, nu2)*t.ppf(left_bracket, df=nu1)

    right_bracket = (np.max([p,alpha]) + 1 - (2*alpha))/(2*(1-alpha))
    right_block = 2*(1-alpha_star(alpha, nu1, nu2))*(t.ppf(right_bracket, df=nu2))

    ## Need to normalize to get back to an unscaled distribution
    ppf = (left_block + right_block)*sigma + mu
    
    return(ppf)

# PDF of the Assymetric student t, cf. Zhu and Galbraith JoE 2010 
def asymt_pdf(y, alpha=0.5, nu1=1, nu2=1, mu=0, sigma=1):
    
    """ 
    Following Zhu and Galbraith, pp. 298 bottom right
    Alpha is the skewness, nu1 and nu2 are the left and right kurtosis
    mu is location (mode) and sigma the scale (variance)
    """
    
    if y <= mu: ## Specify the density on the left tail
        core = (y-mu)/(2*alpha*sigma*K(nu1))
        core2 = np.power(core,2)
        bracket = 1 + (1/nu1)*core2
        bracket_power = np.power(bracket, -(nu1+1)/2)
        pdf = (1/sigma)*bracket_power
        
    else: ## Specify the density on the right tail
        core = (y-mu)/(2*(1-alpha)*sigma*K(nu2))
        core2 = np.power(core, 2)
        bracket = 1 + (1/nu2)*core2
        bracket_power = np.power(bracket, -(nu2+1)/2)
        pdf = (1/sigma)*bracket_power

    return(pdf)

# CDF of the Assymetric student t, cf. Zhu and Galbraith JoE 2010
def asymt_cdf(y_0, alpha=0.5, nu1=1, nu2=1, mu=0, sigma=1):
    
    """ 
    Following Zhu and Galbraith, pp. 299 top left  
    Alpha is the skewness, nu1 and nu2 are the left and right kurtosis
    mu is location (mode) and sigma the scale (variance)
    """
    
    ## Need to normalize so that it works with unscaled version of cdf
    y = (y_0 - mu)/sigma
    
    left_bracket = np.min([y,0])/(2*alpha_star(alpha, nu1, nu2))
    left_block = 2*alpha*t.cdf(left_bracket, df=nu1)

    right_bracket = np.max([y,0])/(2*(1-alpha_star(alpha, nu1, nu2)))
    right_block = 2*(1-alpha)*(t.cdf(right_bracket, df=nu2) - (1/2))

    cdf = left_block + right_block
    
    return(cdf)

# Expectation of the Assymetric student t, cf. Zhu and Galbraith JoE 2010
def asymt_mean(alpha=0.5, nu1=1, nu2=1, mu=0, sigma=1):
    astar=alpha_star_plain(alpha,nu1,nu2)
    knu1=K_plain(nu1)
    knu2=K_plain(nu2)
    #Expectation of standard AST
    East=4*(-alpha*astar*nu1*knu1/(nu1-1)+(1-alpha)*(1-astar)*nu2*knu2/(nu2-1))    
    #Scaled by sigma and shift by mu
    ans=East*sigma+mu
    return ans

# Weighted_kernel_interpolation
class Weighted_kernel:
    """
    A class for performing weighted kernel interpolation for estimating conditional quantiles.

    Parameters
    ----------
    cond_quant : dict
        A dictionary containing the observed conditional quantiles and their corresponding input values. The keys represent the input values (theta) and the values represent the conditional quantiles (q). The dictionary should have hashable keys and values that can be converted to NumPy arrays.

    bandwidth : float or None (default: None), optional
        The smoothing parameter (bandwidth) for the weighted kernel interpolation. If provided, the specified value will be used. If set to None, the bandwidth will be estimated automatically based on the data.
    """
    def __init__(self, cond_quant, bandwidth=None):
            theta = np.array(list(cond_quant.keys()))
            q_values = np.array(list(cond_quant.values()))

            # sort the values based on theta
            sorted_indices = np.argsort(theta)
            theta = theta[sorted_indices]
            q_values = q_values[sorted_indices]

            # estimating the smoothing parameter (bandwidth)
            n = len(q_values)
            if bandwidth:
                h = bandwidth
            else:
                q_std = self.quantile_std(theta, q_values)
                IQR = cond_quant[0.75] - cond_quant[0.25]
                h = 1.06*min(q_std, IQR)*(n**(-1/5))

            # initial inputs and weights
            self.theta = theta
            self.q_values = q_values
            self.bandwidth = h
            self.h = h
            self.w_init = np.ones(n)/n

            # adding special constraint for sum(w)=1
            self.cons = {'type':'eq', 'fun': self.const}

    def _w_kernel_cdf(self, x, w):
        q = self.q_values
        h = self.h
        quant_hats = norm.cdf((x[:, np.newaxis]-q) / h)
        theta_hat = quant_hats @ w
        return theta_hat

    def w_kernel_cdf(self, x):
        """
        Computes the weighted kernel estimate of the cumulative distribution function (CDF) for the given input value(s). The interpolation is performed using the observed conditional quantiles and their weights.

        Parameters
        ----------
        x : scalar, array-like
            The input value(s) for which the CDF estimate is computed. If a scalar, a single CDF estimate is returned. If an array-like object, an array of CDF estimates is returned.
        
        Returns
        -------
        theta_hat : scalar or ndarray
            The estimated quantile(s) corresponding to the input value(s). If a scalar input is given, a single quantile is returned. If an array-like input is given, an array of quantiles is returned.
        """
        q = self.q_values
        h = self.h
        w = self.w_hat
        if np.isscalar(x):
            quant_hats = norm.cdf((x-q) / h)
        else:
            quant_hats = norm.cdf((x[:, np.newaxis]-q) / h)
        theta_hat = quant_hats @ w
        return theta_hat

    def w_kernel_pdf(self, x):
        """
        Computes the weighted kernel estimate of the probability density function (PDF) for the given input value(s). The interpolation is performed using the observed conditional quantiles and their weights.

        Parameters
        ----------
        x : scalar, array-like
            The input value(s) for which the PDF estimate is computed. If a scalar, a single PDF estimate is returned. If an array-like object, an array of PDF estimates is returned.
        
        Returns
        -------
        theta_hat : scalar or ndarray
            The estimated PDF value(s) corresponding to the input value(s). If a scalar input is given, a single PDF value is returned. If an array-like input is given, an array of PDF values is returned.
        """
        q = self.q_values
        h = self.h
        w = self.w_hat
        if np.isscalar(x):
            quant_hats = norm.pdf((x-q) / h)
        else:
            quant_hats = norm.pdf((x[:, np.newaxis]-q) / h)
        theta_hat = (quant_hats @ w) / h
        return theta_hat
    
    def guess_ppf(self, theta):
        is_scalar = np.isscalar(theta)
        if is_scalar:
            theta = np.array([theta])
        neighbour_indices = np.argsort(np.abs(theta[:, np.newaxis] - self.theta))[:, :2]
        slopes = (np.diff(self.q_values[neighbour_indices]) / np.diff(self.theta[neighbour_indices])).reshape(-1)
        intercepts = self.q_values[neighbour_indices][:, 0] - self.theta[neighbour_indices][:, 0] * slopes
        q_values = intercepts + theta * slopes
        if is_scalar:
            q_values = q_values[0]
        return q_values
        
    def w_kernel_ppf(self, theta):
        q_guess = self.guess_ppf(theta)
        if np.isscalar(theta):
            return root(lambda x: self.w_kernel_cdf(x) - theta, q_guess)['x'][0]
        else:
            return root(lambda x: self.w_kernel_cdf(x) - theta, q_guess)['x']

    def w_kernel_loss(self, w):
        theta = self.theta
        q = self.q_values
        theta_hat = self._w_kernel_cdf(q, w)
        main_loss = np.sum(np.power(theta - theta_hat, 2))
        total_loss = main_loss
        return total_loss
   
    @staticmethod
    def const(x):
        return x.sum() - 1

    @staticmethod
    def moving_average(a, n=2) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def quantile_std(self, theta, q):
        density = np.diff(theta)/np.diff(q)
        norm_density = density/np.sum(density)
        q_v = self.moving_average(q, 2)
        q_mean = q_v @ norm_density
        q_std = np.sqrt(np.power(q_v - q_mean, 2) @ norm_density)
        return q_std

    def w_kernel_fit(self):
        res = minimize(self.w_kernel_loss, x0=self.w_init, 
                       method='SLSQP', bounds=[(0, None)],
                       constraints=self.cons, 
                       options={'maxiter':100})
        self.w_hat = res.x
        return self.w_hat

def gen_PDF_and_CDF(model_fit, fittype, x_list=None, loc=None):
    if fittype=='T-skew':
        dfpdf = gen_t_PDF_and_CDF(model_fit, x_list)
    elif fittype=='Asymmetric T':
        dfpdf = gen_asymt_PDF_and_CDF(model_fit, loc, x_list)
    elif fittype=='Kernel-based':
        dfpdf = gen_kernel_PDF_and_CDF(model_fit, x_list)
    return dfpdf

def gen_t_PDF_and_CDF(tsfit, x_list=None):
    if type(x_list)==type(None):
        v_q5 = tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        v_q40 = tskew_ppf(0.4, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        v_q60 = tskew_ppf(0.6, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        v_q95 = tskew_ppf(0.95, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        min_v = v_q5-abs(v_q5-v_q40)
        max_v = v_q95+abs(v_q95-v_q60)
        while tskew_cdf(min_v+1, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])>0.05:
            min_v-=1
        x_list = [x for x in np.arange(min_v,max_v,0.05)]
    yvals = [tskew_pdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]    
    ycdf = [tskew_cdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]
    tmp_dic={'Tskew_PDF_x':x_list,'Tskew_PDF_y':yvals,'Tskew_CDF':ycdf}
    dfpdf = pd.DataFrame(tmp_dic)
    return dfpdf

def gen_asymt_PDF_and_CDF(asymtfit, loc, x_list=None):
    min_v = loc-1
    max_v = loc+1
    while asymt_cdf(min_v+0.2, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])>0.05:
        min_v-=0.2
    if type(x_list)==type(None):
        x_list = [x for x in np.arange(min_v,max_v,0.02)]
    yvals= [asymt_pdf(z, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) for z in x_list]    
    ycdf = [asymt_cdf(z, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) for z in x_list]
    
    tmp_dic = {'AsymT_PDF_x':x_list,'AsymT_PDF_y':yvals,'AsymT_CDF':ycdf}
    dfpdf = pd.DataFrame(tmp_dic)
    return dfpdf

def gen_kernel_PDF_and_CDF(kernel_model, x=None):
    if type(x)==type(None):
        v_q1, v_q5, v_q40, v_q60, v_q95, v_q99 = kernel_model.w_kernel_ppf(np.array([0.01, 0.05, 0.4, 0.6, 0.95, 0.99]))
        min_v = max(v_q5-abs(v_q5-v_q40), v_q1)
        max_v = min(v_q95+abs(v_q95-v_q60), v_q99)
        x = np.array([x for x in np.linspace(min_v,max_v,500)])
    density_hat = kernel_model.w_kernel_pdf(x)
    theta_hat = kernel_model.w_kernel_cdf(x)
    dfpdf = pd.DataFrame({'Kernel_PDF_x':x, 'Kernel_PDF_y':density_hat,
                          'Kernel_CDF_y':theta_hat})
    return dfpdf

def select_t_x_list(tsfits):
    if len(tsfits)==0:
        return []
    loclist = [tsfit['loc'] for tsfit in tsfits]
    min_v = min(loclist)-8
    max_v = max(loclist)+8
    for tsfit in tsfits:
        v_q15=tskew_ppf(0.15, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        v_q40=tskew_ppf(0.4, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        v_q60=tskew_ppf(0.6, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        v_q85=tskew_ppf(0.85, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

        # increase the range if some quantiles are outside
        min_v = min(min_v,v_q15-abs(v_q15-v_q40))
        max_v = max(max_v,v_q85+abs(v_q85-v_q60))
    x_list = np.array([x for x in np.linspace(min_v,max_v,500)])
    return x_list

def select_asymt_x_list(asymtfits):
    if len(asymtfits)==0:
        return []
    loclist = [asymtfit['loc'] for asymtfit in asymtfits]
    min_v = min(loclist)-1.5
    max_v = max(loclist)+1.5

    for asymtfit in asymtfits:
        v_q15 = asymt_ppf(0.15, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
        v_q40 = asymt_ppf(0.4, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
        v_q60 = asymt_ppf(0.6, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
        v_q85 = asymt_ppf(0.85, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])

        # increase the range if some quantiles are outside
        min_v = min(min_v,v_q15-abs(v_q15-v_q40))
        max_v = max(max_v,v_q85+abs(v_q85-v_q60))
    x_list = np.array([x for x in np.linspace(min_v,max_v,500)])
    return x_list

def select_kernel_x_list(kfits):
    if len(kfits)==0:
        return []
    # estimate the mean for each period
    meanlist = []
    for kfit in kfits:
        x = kfit.q_values
        ypdf = kfit.w_kernel_pdf(x)
        meanx = (x @ ypdf)/np.sum(ypdf)
        meanlist.append(meanx)

    # set initial values
    min_v = min(meanlist)-8
    max_v = max(meanlist)+8

    for kfit in kfits:
        v_q15, v_q40, v_q60, v_q85 = kfit.w_kernel_ppf(np.array([0.15, 0.4, 0.6, 0.85]))

        # increase the range if some quantiles are outside
        min_v = min(min_v,v_q15-abs(v_q15-v_q40))
        max_v = max(max_v,v_q85+abs(v_q85-v_q60))
    x_list = np.array([x for x in np.linspace(min_v,max_v,500)])
    return x_list

def select_x_list(model_fits, methods):
    tskew_model_fits = [model_fit for method, model_fit in zip(methods, model_fits) if method=='T-skew']
    asymt_model_fits = [model_fit for method, model_fit in zip(methods, model_fits) if method=='Asymmetric T']
    kernel_model_fits = [model_fit for method, model_fit in zip(methods, model_fits) if method=='Kernel-based']

    tskew_x_list = select_t_x_list(tskew_model_fits)
    asymt_x_list = select_asymt_x_list(asymt_model_fits)
    kernel_x_list = select_kernel_x_list(kernel_model_fits)
    xs = [tskew_x_list, asymt_x_list, kernel_x_list]
    xs = [x for x in xs if len(x)!=0]
    min_v, max_v = min(xs[0]), max(xs[0])
    for x in xs:
        min_i, max_i = min(x), max(x)
        min_v = min_i if min_i < min_v else min_v
        max_v = max_i if max_i > max_v else max_v
    x_list = np.array([x for x in np.linspace(min_v,max_v,500)])
    return x_list
