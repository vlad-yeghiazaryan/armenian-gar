## 3rd-party modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .partition import retropolated_PCA
from .tsfit import tskew_fit, asymt_fit, kernel_fit
from .tsfit import gen_t_PDF_and_CDF, gen_asymt_PDF_and_CDF

# Functions for step 4: scenario test
def run_scenario(fitdate, cond_quant_raw, cond_quant_shocked, fitparam, fitparam_shocked, horizon, **kwargs):
    '''
    Main run function for step 2, quantfit.

    Takes in as arguments a dict for input parameters
    and a df for data. Outputs a dict for output parameters.

    Does quantile fits and returns a dict of output parameters.
    ** This function should be independent of any Excel input/output
    and be executable as a regular Python function independent of Excel. **
    '''
    # ------------------------
    # Create output dict
    # ------------------------
    dict_output_scenario = dict()
    ols_raw = cond_quant_raw.pop('mean')
    ols_shocked = cond_quant_shocked.pop('mean')
   
    fig, res, dfpdf= scenario_compare(cond_quant_raw, cond_quant_shocked,fitparam, fitparam_shocked,horizon,fitdate, ols_raw, ols_shocked)
    dict_output_scenario['fig'] = fig
    dict_output_scenario['res'] = res
    dict_output_scenario['dfpdf'] = dfpdf

    return dict_output_scenario

# Calculate shock relations
def gen_relation(shockdict, partition_groups, original_data, df_partition, target):
    df_shockedvar = pd.DataFrame(index=original_data.index)
    df_shockedgrp = df_partition.copy()
    for group in df_shockedgrp.columns:
        if group not in ['date', target]:
            df_shockedgrp[group+'_shocked']=df_shockedgrp[group]
    for var, shock in shockdict.items():
        ct=0
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
            if var in compvars and var in partition_groups.keys():
                print(var+' is not well  defined.')
                
            if var in compvars:
                ct+=1
                df_var=original_data[['date',var]].dropna()
                df_part=df_partition[['date',group]].dropna()

                sdate=max(min(df_var['date'].values),(min(df_part['date'].values)))
                edate=min(max(df_var['date'].values),(max(df_part['date'].values)))

                df_var=df_var[(df_var['date']>=sdate) & (df_var['date']<=edate)]
                df_part=df_part[(df_part['date']>=sdate) & (df_part['date']<=edate)]
                
                # use correlation to understand how much of the shock will leak into the group 
                cov=np.corrcoef(df_var[var].values,df_part[group].values)[0][1]
                if shock['shocktype']=='By +/- STD':
                    # !!! Will have to review the logic behind this !!!
                    df_shockedgrp[group+'_shocked']= df_shockedgrp[group+'_shocked']+std*shock['shockvalue']*cov
                elif shock['shocktype']=='By +/- percentage':
                    df_shockedgrp[group+'_shocked']= df_shockedgrp[group+'_shocked']+original_data[var]*shock['shockvalue']*cov               
            elif var==group:
                ct+=1
                print(group,var,shock['shocktype'],shock['shockvalue'])
                if shock['shocktype']=='By +/- STD':
                    df_shockedgrp[group+'_shocked']= df_shockedgrp[group+'_shocked']+std*shock['shockvalue']
                elif shock['shocktype']=='By +/- percentage':                
                    df_shockedgrp[group+'_shocked']= df_shockedgrp[group+'_shocked']+df_shockedgrp[group]*shock['shockvalue']

        if ct==0:
            print(var+' not in any group.')
    return df_shockedvar, df_shockedgrp

def gen_shocked_PCA(shockvars, partition_groups, original_data, transformer, target, horizon=4, method_growth='cpd', method='PCA', benchcutoff=0.2):
    # perform PCA
    df_partition, partition_load = retropolated_PCA(original_data, partition_groups, target, horizon, method_growth, method, benchcutoff)

    # generate some shocks
    df_shockedvar, df_shockedgrp = gen_relation(shockvars, partition_groups, original_data, df_partition, target)
    
    # select the shocked columns only
    shock_cols = [c for c in df_shockedgrp.columns if (c not in df_partition.columns) or (c in ['date', target])]
    df_shockedgrp = df_shockedgrp[shock_cols].copy()

    if type(transformer)!=type(None):
        # apply transformations after shock generation
        transformer.mapping = {c:c+'_shocked' for c in list(transformer.mapping.keys()) + list(partition_groups.keys())}
        df_shockedgrp = transformer.transform(df_shockedgrp, target)

        # selected shocked columns
        com_cols = list(set(transformer.mapping.values()) & set(df_shockedgrp.columns))
        df_shockedgrp = df_shockedgrp[['date']+com_cols]
        transformer.mapping = {c:c for c, s in transformer.mapping.items()}

    # set index for shockvar
    df_shockedvar.index = original_data['date']
    return df_shockedvar, df_shockedgrp

def scenario_compare(cond_quant_raw, cond_quant_shocked, fitparam,fitparam_shocked, horizon, fitdate, ols_raw, ols_shocked):
    if fitparam['fittype']=='T-skew':
        tsfit_raw = tskew_fit(cond_quant_raw, fitparam)
        tsfit_shocked = tskew_fit(cond_quant_shocked, fitparam_shocked)
        loc_raw = tsfit_raw['loc']
        loc_shocked = tsfit_shocked['loc'] 

        dfpdf_shocked = gen_t_PDF_and_CDF(tsfit_shocked)
        dfpdf_raw = gen_t_PDF_and_CDF(tsfit_raw, x_list=dfpdf_shocked['Tskew_PDF_x'])

        tmp_dic={
            'Tskew_PDF_x':dfpdf_shocked['Tskew_PDF_x'],'Tskew_PDF_y_before':dfpdf_raw['Tskew_PDF_y'],'Tskew_CDF_y_before':dfpdf_raw['Tskew_CDF'],'Tskew_PDF_y_after':dfpdf_shocked['Tskew_PDF_y'],'Tskew_CDF_y_after':dfpdf_shocked['Tskew_CDF']
        }
        dfpdf=pd.DataFrame(tmp_dic)
        x_list = dfpdf['Tskew_PDF_x']

        for i,y in enumerate(dfpdf['Tskew_CDF_y_before']):
            if y>0.05:
                q5loc_raw=i
                break
        for i,y in enumerate(dfpdf['Tskew_CDF_y_before']):
            if y>0.1:
                q10loc_raw=i
                break
        
        for i,y in enumerate(dfpdf['Tskew_CDF_y_after']):
            if y>0.05:
                q5loc_shocked=i
                break
        for i,y in enumerate(dfpdf['Tskew_CDF_y_after']):
            if y>0.1:
                q10loc_shocked=i
                break
        fig = plot_T_dist(fitdate, dfpdf, q5loc_raw, q5loc_shocked, horizon)
        res=[]
        res.append([' ','Before shock','After shock'])
        res.append(['Date of input',fitdate,fitdate])
        res.append(['Horizon forward',horizon,horizon])
        res.append(['Conditional mode',loc_raw,loc_shocked])
        res.append(['GaR5%',x_list[q5loc_raw-1],x_list[q5loc_shocked-1]])
        res.append(['GaR10%',x_list[q10loc_raw-1],x_list[q10loc_shocked-1]])
        res.append(['Skewness',tsfit_raw['skew'],tsfit_shocked['skew']])
        res.append(['Scale',tsfit_raw['scale'],tsfit_shocked['scale']])   
        return fig, res, dfpdf

    elif fitparam['fittype']=='Asymmetric T':    
        asfit_raw=asymt_fit(cond_quant_raw, fitparam, ols_raw)
        asfit_shocked=asymt_fit(cond_quant_shocked, fitparam_shocked,ols_shocked)
        loc_raw=asfit_raw['loc']
        loc_shocked=asfit_shocked['loc']

        dfpdf_raw = gen_asymt_PDF_and_CDF(asfit_raw, loc_raw)
        dfpdf_shocked = gen_asymt_PDF_and_CDF(asfit_shocked, loc_shocked)

        tmp_dic={
            'AsymT_PDF_x':dfpdf_shocked['AsymT_PDF_x'],'AsymT_PDF_y_before':dfpdf_raw['AsymT_PDF_y'],'AsymT_CDF_y_before':dfpdf_raw['AsymT_CDF_y'],'AsymT_PDF_y_after':dfpdf_shocked['AsymT_PDF_y'],'AsymT_CDF_y_after':dfpdf_shocked['AsymT_CDF_y']
        }
        dfpdf=pd.DataFrame(tmp_dic)
        x_list = dfpdf['AsymT_PDF_x']

        for i,y in enumerate(dfpdf['AsymT_CDF_y_before']):
            if y>0.05:
                q5loc_raw=i
                break
        for i,y in enumerate(dfpdf['AsymT_CDF_y_before']):
            if y>0.1:
                q10loc_raw=i
                break
        
        for i,y in enumerate(dfpdf['AsymT_CDF_y_after']):
            if y>0.05:
                q5loc_shocked=i
                break
        for i,y in enumerate(dfpdf['AsymT_CDF_y_after']):
            if y>0.1:
                q10loc_shocked=i
                break
        
        fig = plot_asymt_T_dist(fitdate, dfpdf, q5loc_raw, q5loc_shocked, horizon)
        res=[]
        res.append([' ','Before shock','After shock'])
        res.append(['Date of input',fitdate,fitdate])
        res.append(['Horizon forward',horizon,horizon])
        res.append(['Conditional mode',loc_raw,loc_shocked])
        res.append(['GaR5%',x_list[q5loc_raw-1],x_list[q5loc_shocked-1]])
        res.append(['GaR10%',x_list[q10loc_raw-1],x_list[q10loc_shocked-1]])
        res.append(['Skew parameter',asfit_raw['skew'],asfit_shocked['skew']])
        res.append(['Scale',asfit_raw['scale'],asfit_shocked['scale']])   
        return fig, res, dfpdf

    elif fitparam['fittype']=='Kernel-based':
        fig, res, dfpdf = gen_kernel_comparison(cond_quant_raw, cond_quant_shocked, fitparam,fitparam_shocked, horizon, fitdate)
        return fig, res, dfpdf

def gen_kernel_comparison(cond_quant_raw, cond_quant_shocked, fitparam,fitparam_shocked, horizon, fitdate):
    res_raw, dfpdf_raw = kernel_fit(cond_quant_raw, fitparam, fitdate, horizon)
    x_raw, ypdf_raw, ycdf_raw = dfpdf_raw.values.T
    res_shocked, dfpdf_shocked = kernel_fit(cond_quant_shocked, fitparam_shocked, fitdate, horizon, x_raw)
    x_shocked, ypdf_shocked, ycdf_shocked = dfpdf_shocked.values.T
    tmp_dic={
        'Tskew_PDF_x':x_raw, 'Tskew_PDF_y_before':ypdf_raw,
        'Tskew_CDF_y_before':ycdf_raw, 'Tskew_PDF_y_after':ypdf_shocked,'Tskew_CDF_y_after':ycdf_shocked
    }
    dfpdf = pd.DataFrame(tmp_dic)
    res_raw['scenario'] = 'before shock'
    res_shocked['scenario'] = 'after shock'
    res = pd.DataFrame([res_raw, res_shocked])
    fig = plot_kernel_dist_comparison(fitdate, horizon, dfpdf_raw, dfpdf_shocked)   
    return fig, res, dfpdf

def plot_T_dist(fitdate, dfpdf, q5loc_raw, q5loc_shocked, horizon):
    x_list = dfpdf['Tskew_PDF_x']
    yvals_raw = dfpdf['Tskew_PDF_y_before']
    yvals_shocked = dfpdf['Tskew_PDF_y_after']

    titlestr = "Scenario test for "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    lablestr_raw = "Density before shock"
    lablestr_shocked = "Density after shock"
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.set_title(titlestr,fontsize=24)

    if yvals_raw[q5loc_raw]>yvals_shocked[q5loc_raw]:
        ax.fill_between(x_list[:q5loc_raw], 0, yvals_raw[:q5loc_raw],  facecolor='c', interpolate=True)
        ax.fill_between(x_list[:q5loc_shocked], 0, yvals_shocked[:q5loc_shocked],  facecolor='g', interpolate=True)
    else:
        ax.fill_between(x_list[:q5loc_shocked], 0, yvals_shocked[:q5loc_shocked],  facecolor='g', interpolate=True)
        ax.fill_between(x_list[:q5loc_raw], 0, yvals_raw[:q5loc_raw],  facecolor='c', interpolate=True)

    ax.plot(x_list,yvals_raw,'c-',label=lablestr_raw)
    ax.plot(x_list,yvals_shocked,'g-',label=lablestr_shocked)
    ax.legend(fontsize=24)
    ax.tick_params(labelsize=24)
    plt.ylim(0, max(max(yvals_raw),max(yvals_shocked))*1.2)
    plt.ylabel('Probability Density', fontsize=24)
    plt.xlabel('GDP (compound annual growth rate)', fontsize=24)  
    plt.close('all')
    return fig

def plot_asymt_T_dist(fitdate, dfpdf, q5loc_raw, q5loc_shocked, horizon):
    x_list = dfpdf['AsymT_PDF_x']
    yvals_raw = dfpdf['AsymT_PDF_y_before']
    yvals_shocked = dfpdf['AsymT_PDF_y_after']

    titlestr = "Scenario test for "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    lablestr_raw = "Density before shock"
    lablestr_shocked = "Density after shock"
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.set_title(titlestr,fontsize=24)

    if yvals_raw[q5loc_raw]>yvals_shocked[q5loc_raw]:
        ax.fill_between(x_list[:q5loc_raw], 0, yvals_raw[:q5loc_raw],  facecolor='c', interpolate=True)
        ax.fill_between(x_list[:q5loc_shocked], 0, yvals_shocked[:q5loc_shocked],  facecolor='g', interpolate=True)
    
    else:
        ax.fill_between(x_list[:q5loc_shocked], 0, yvals_shocked[:q5loc_shocked],  facecolor='g', interpolate=True)
        ax.fill_between(x_list[:q5loc_raw], 0, yvals_raw[:q5loc_raw],  facecolor='c', interpolate=True) 
    ax.plot(x_list,yvals_raw,'c-',label=lablestr_raw)
    ax.plot(x_list,yvals_shocked,'g-',label=lablestr_shocked)
    ax.legend(fontsize=24)
    ax.tick_params(labelsize=24)
    plt.ylim(0, max(max(yvals_raw),max(yvals_shocked))+0.2)
    plt.ylabel('Probability Density', fontsize=24)
    plt.xlabel('GDP (compound annual growth rate)', fontsize=24)
    plt.close('all')
    return fig

def plot_kernel_dist_comparison(fitdate, horizon, dfpdf_raw, dfpdf_shocked):
    # setting plot inputs
    x_raw, ypdf_raw, ycdf_raw = dfpdf_raw.values.T
    x_shocked, ypdf_shocked, ycdf_shocked = dfpdf_shocked.values.T
    q5loc_raw = np.argmin(np.abs(ycdf_raw - 0.05))
    q5loc_shocked = np.argmin(np.abs(ycdf_shocked - 0.05))
    q10loc_raw = np.argmin(np.abs(ycdf_raw - 0.1))
    q10loc_shocked = np.argmin(np.abs(ycdf_shocked - 0.1))

    # plot text inputs
    titlestr = "Scenario test for "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    lablestr_raw = "Density before shock"
    lablestr_shocked = "Density after shock"

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.set_title(titlestr,fontsize=24)

    # fill the smaller tail of the plot last so that both are visible
    if ypdf_raw[q5loc_raw]>ypdf_shocked[q5loc_raw]:
        ax.fill_between(x_raw[:q5loc_raw], 0, ypdf_raw[:q5loc_raw],  facecolor='c', interpolate=True)
        ax.fill_between(x_shocked[:q5loc_shocked], 0, ypdf_shocked[:q5loc_shocked],  facecolor='g', interpolate=True)
    else:
        ax.fill_between(x_raw[:q5loc_shocked], 0, ypdf_shocked[:q5loc_shocked],  facecolor='g', interpolate=True)
        ax.fill_between(x_raw[:q5loc_raw], 0, ypdf_raw[:q5loc_raw],  facecolor='c', interpolate=True)
    
    # plot the PDF
    ax.plot(x_raw, ypdf_raw,'c-',label=lablestr_raw)
    ax.plot(x_shocked, ypdf_shocked,'g-',label=lablestr_shocked)
    ax.legend(fontsize=24)
    ax.tick_params(labelsize=24)
    plt.ylim(0, max(max(ypdf_raw),max(ypdf_shocked))*1.2)
    plt.ylabel('Probability Density', fontsize=24)
    plt.xlabel('GDP (compound annual growth rate)', fontsize=24)  
    plt.close('all')
    return fig
