## 3rd-party modules
import pandas as pd
import numpy as np

from .tsfit import tskew_pdf
from .tsfit import tskew_cdf
from .tsfit import tskew_ppf
from .tsfit import asymt_pdf
from .tsfit import asymt_cdf
from .tsfit import asymt_ppf
from .tsfit import tskew_fit
from .tsfit import asymt_fit, tskew_ppf_vec, gen_t_PDF_and_CDF, gen_asymt_PDF_and_CDF, asymt_ppf_vec, get_cond_quant
from .tsfit import quantile_uncrossing, Weighted_kernel, gen_kernel_values

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import log

# Function for step 4: historical test
def run_historical(dict_input_historical, data, qcoef):
    # ------------------------
    # Create output dict
    # ------------------------
    dict_output_historical = {}
    target = dict_input_historical['target']
    horizon = dict_input_historical['horizon']
    sdate=dict_input_historical['start_date']
    edate=dict_input_historical['end_date']
    time_inc = dict_input_historical['time_inc']
    fitparam=dict_input_historical['fit_params']

    dates, realvalues, olsmeans, cond_quants = get_cond_quants(sdate, edate, time_inc, data, qcoef, target, horizon, fitparam)
    figs, res, chartpacks = historical_gen(cond_quants, fitparam, realvalues, olsmeans, dates, horizon)
    df=pd.DataFrame(res, index=dates)
    cond_quants = pd.DataFrame(cond_quants, index=dates)
    dict_output_historical['figs'] = figs
    dict_output_historical['charts'] = chartpacks
    dict_output_historical['data'] = df
    dict_output_historical['cond_quants'] = cond_quants
    return dict_output_historical

def get_cond_quants(sdate, edate, time_inc, data, qcoef, target, horizon, fitparam):
    # selected the dates for estimating conditional quantiles
    y = data.set_index('date')[target]
    dates = data['date'][(data['date']>=sdate) & (data['date']<=edate)]
    dates = dates.iloc[list(range(0,len(dates),time_inc))].values
    
    cond_quants=[]
    realvalues=[]
    olsmeans=[]
    fitted_dates=[]
    for fitdate in dates:
        cond_quant = get_cond_quant(fitdate, data, qcoef, target, horizon, fitparam['qsmooth'], fitparam['qsmooth_period'])
        olsmean = cond_quant.pop('mean')
        realvalue = y.loc[fitdate]

        # skip in cases where fitted values for the dates are missing
        if pd.isna(olsmean):
            continue
        olsmeans.append(olsmean)
        cond_quants.append(cond_quant)
        realvalues.append(realvalue)
        fitted_dates.append(fitdate)
    fitted_dates = np.array(fitted_dates)
    return fitted_dates, realvalues, olsmeans, cond_quants

def historical_gen(cond_quants, fitparam, realvalues, olsmeans, dates, horizon):
    n = len(cond_quants)
    n_charts=10
    draws=list(range(n))
    if n>n_charts:
        draws=[int(n*i/n_charts) for i in range(n_charts)]
    if fitparam['fittype']=='T-skew':
        res, tsfits = get_hist_T_dist(cond_quants, fitparam)
        figs, chartpacks = plot_hist_T_dist(dates, res, tsfits, realvalues, draws)
        return figs, res, chartpacks
    elif fitparam['fittype']=='Asymmetric T':
        res, asfits = get_hist_asymt_dist(cond_quants, fitparam, olsmeans)
        figs, chartpacks = plot_hist_asymt_dist(dates, res, asfits, realvalues, draws)
        return figs, res, chartpacks
    elif fitparam['fittype']=='Kernel-based':
        res = get_hist_kernel_dist(cond_quants, fitparam, dates, horizon)
        return {}, res, []

def get_hist_T_dist(cond_quants, fitparam):
    res = []
    tsfits=[]
    quantiles = list(cond_quants[-1].keys())

    # perform fitting
    for cond_quant in cond_quants:
        tsfit = tskew_fit(cond_quant, fitparam)
        tsfits.append(tsfit)
    
    # select a fixed x_list 
    x_list = select_t_x_list(tsfits)

# ToDo: !!! this code/for loop takes too long to run !!!
    for tsfit in tsfits:
        # save fitted values
        res_fit = {
            'location': tsfit['loc']/tsfit['scale'],
            'scale': tsfit['scale'],
            'skew': tsfit['skew']
        }
        # store PDF and CDF
        res_fit['dfpdf'] = gen_t_PDF_and_CDF(tsfit, x_list)

        # store values for different tails
        quantiles.append(0.05) if 0.05 not in quantiles else None
        q_values = tskew_ppf_vec(quantiles, tsfit)
        for q, v in q_values.items():
            res_fit['var'+str(int(q*100))+'%'] = v
        res.append(res_fit)
    return res, tsfits

def plot_hist_T_dist(dates, res, tsfits, realvalues, draws):
    chartpacks=[]
    pits=[]
    logscore={}
    logscore['uncensored']=[]
    logscore['10tail']=[]
    logscore['90tail']=[]

    for ct, tsfit in enumerate(tsfits):
        x_list = res[ct]['dfpdf']['Tskew_PDF_x']
        if not np.isnan(realvalues[ct]):
            realcdf=tskew_cdf(realvalues[ct], df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            realpdf=tskew_pdf(realvalues[ct], df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        else:
            realcdf=tskew_cdf(tsfit['loc']/tsfit['scale'], df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            realpdf=tskew_pdf(tsfit['loc']/tsfit['scale'], df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        pits.append(realcdf)
        logscore['uncensored'].append(log(realpdf))
        if realcdf<=0.1:
            logscore['10tail'].append(log(realpdf))
        else:
            logscore['10tail'].append(log(0.9))
        if realcdf>=0.9:
            logscore['90tail'].append(log(realpdf))
        else:
            logscore['90tail'].append(log(0.9))

        if ct in draws:
            figchart, ax = plt.subplots(1, 1, figsize=(10,5))
            yvals= [tskew_pdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list] 
        
            titlestr = " T-skew quantile fit for "+str(dates[ct])[:10]+" growth rate"
            lablestr = "Density "+str(dates[ct])[:10]+" "+"growth rate"
            ax.plot(x_list,yvals,'b-',label=lablestr)
            if np.isnan(realvalues[ct]):
                modx=res[-1]['location']
            else:
                modx=realvalues[ct]
            mody=max(yvals)
            ax.plot([modx,modx],[0,mody],'r-.')
            ax.set_title(titlestr)
            ax.legend()
            chartpacks.append(figchart)           
    
    pits_cdf=[]
    npits=len(pits)
    for r in np.arange(0,1,0.01):
        pits_cdf.append(len([x for x in pits if x<=r])/npits) # Calculate how many realized values are below any given probability
    
    figpit, axpit= plt.subplots(1, 1, figsize=(8,8))
    axpit.plot(list(np.arange(0,1,0.01)),pits_cdf,'r-',label='Realized')
    axpit.plot(list(np.arange(0,1,0.01)),list(np.arange(0,1,0.01)),'b-',label='U~(0,1)')
    axpit.plot(list(np.arange(0,1,0.01)),[e+1.34*npits**(-0.5) for e in np.arange(0,1,0.01)],'b--',label='5 percent critical values')
    axpit.plot(list(np.arange(0,1,0.01)),[e-1.34*npits**(-0.5) for e in np.arange(0,1,0.01)],'b--')
    axpit.set_title('Probablity inversion test', fontsize=16, y=1.01)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(loc=4,fontsize=12)

    figls, axls= plt.subplots(1, 1, figsize=(12,8))
    axls.plot(dates, logscore['uncensored'], 'k-', label='Uncensored log score')
    axls.plot(dates, logscore['10tail'], 'r--',label='Censored log score for left  10% tail')
    axls.plot(dates, logscore['90tail'], 'g-.', label='Censored log score for right 10% tail')
    plt.legend(loc=3,fontsize=12)
    plt.title('Uncensored and Censored Logscores',fontsize=16,y=1.01)
    
    res = pd.DataFrame(res)
    para=['var10%','location','scale','skew']
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15,80))
    
    for i,k in enumerate (para):
        axes[i].plot(dates,res[k])
        axes[i].set_title('Historical distribution of T-skew parameter: {}'.format(k), fontsize=24, y=1.03)
        axes[i].tick_params(labelsize=16)
        axes[i].set_xlabel('')
    figs={}
    figs['res']=fig
    figs['pit']=figpit
    figs['ls']=figls
    plt.close('all')
    return figs, chartpacks

def get_hist_asymt_dist(cond_quants, fitparam, olsmeans):
    res = []
    asfits=[]
    quantiles = list(cond_quants[-1].keys())

    # perform fitting
    for i, cond_quant in enumerate(cond_quants):
        asfit = asymt_fit(cond_quant, fitparam, olsmeans[i])
        asfits.append(asfit)
    
    # select a fixed x_list 
    x_list = select_asymt_x_list(asfits)
    for asfit in asfits:
        # save fitted values
        res_fit = {
            'location': asfit['loc']/asfit['scale'],
            'scale': asfit['scale'],
            'skew': asfit['skew']
        }
        # store PDF and CDF
        res_fit['dfpdf'] = gen_asymt_PDF_and_CDF(asfit, asfit['loc'], x_list)

        # store values for different tails
        quantiles.append(0.05) if 0.05 not in quantiles else None
        q_values = asymt_ppf_vec(quantiles, asfit)
        for q, v in q_values.items():
            res_fit['var'+str(int(q*100))+'%'] = v
        res.append(res_fit)
    return res, asfits

def plot_hist_asymt_dist(dates, res, asfits, realvalues, draws):
    chartpacks=[]
    pits=[]
    logscore={}
    logscore['uncensored']=[]
    logscore['10tail']=[]
    logscore['90tail']=[]

    for ct, asfit in enumerate(asfits):
        x_list = res[ct]['dfpdf']['AsymT_PDF_x']
        if not np.isnan(realvalues[ct]):
            realcdf=asymt_cdf(realvalues[ct], alpha=asfit['skew'], nu1=asfit['kleft'], nu2=asfit['kright'], mu=asfit['loc'], sigma=asfit['scale'])
            realpdf=asymt_pdf(realvalues[ct],alpha=asfit['skew'], nu1=asfit['kleft'], nu2=asfit['kright'], mu=asfit['loc'], sigma=asfit['scale'])
        else:
            realcdf=asymt_cdf(asfit['loc'], alpha=asfit['skew'], nu1=asfit['kleft'], nu2=asfit['kright'], mu=asfit['loc'], sigma=asfit['scale'])
            realpdf=asymt_pdf(asfit['loc'], alpha=asfit['skew'], nu1=asfit['kleft'], nu2=asfit['kright'], mu=asfit['loc'], sigma=asfit['scale'])
        pits.append(realcdf)
        logscore['uncensored'].append(log(realpdf))
        if realcdf<=0.1:
            logscore['10tail'].append(log(realpdf))
        else:
            logscore['10tail'].append(log(0.9))
        if realcdf>=0.9:
            logscore['90tail'].append(log(realpdf))
        else:
            logscore['90tail'].append(log(0.9))
            
        if ct in draws:
            figchart, ax = plt.subplots(1, 1, figsize=(10,5))
            yvals= [asymt_pdf(z, alpha=asfit['skew'], nu1=asfit['kleft'], nu2=asfit['kright'], mu=asfit['loc'], sigma=asfit['scale'])for z in x_list] 
        
            titlestr = " Asymmetric T quantile fit for "+str(dates[ct])[:10]+" growth rate"
            lablestr = "Density "+str(dates[ct])[:10]+" "+"growth rate"
            ax.plot(x_list,yvals,'b-',label=lablestr)
            if np.isnan(realvalues[ct]):
                modx=res['location'][-1]
            else:
                modx=realvalues[ct]
            mody=max(yvals)
            ax.plot([modx,modx],[0,mody],'r-.')
            ax.set_title(titlestr)
            ax.legend()
            chartpacks.append(figchart)
            
    pits_cdf=[]        
    npits=len(pits)
    for r in np.arange(0,1,0.01):
        pits_cdf.append(len([x for x in pits if x<=r])/npits) # Calculate how many realized values are below any given probability
    
    figpit, axpit= plt.subplots(1, 1, figsize=(8,8))
    axpit.plot(list(np.arange(0,1,0.01)),pits_cdf,'r-',label='Realized')
    axpit.plot(list(np.arange(0,1,0.01)),list(np.arange(0,1,0.01)),'b-',label='U~(0,1)')
    axpit.plot(list(np.arange(0,1,0.01)),[e+1.34*npits**(-0.5) for e in np.arange(0,1,0.01)],'b--',label='5 percent critical values')
    axpit.plot(list(np.arange(0,1,0.01)),[e-1.34*npits**(-0.5) for e in np.arange(0,1,0.01)],'b--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(loc=4)
    plt.title('Probablity inversion test.')
    
    figls, axls= plt.subplots(1, 1, figsize=(12,8))
    axls.plot(dates, logscore['uncensored'], 'k-', label='Uncensored log score')
    axls.plot(dates, logscore['10tail'], 'r--',label='Censored log score for left  10% tail')
    axls.plot(dates, logscore['90tail'], 'g-.', label='Censored log score for right 10% tail')
    plt.legend(loc=3,fontsize=12)
    plt.title('Uncensored and Censored Logscores',fontsize=16,y=1.01)
    
    para=['var10%','location','scale','skew']
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15,80))
    
    res = pd.DataFrame(res)
    for i,k in enumerate (para):
        axes[i].plot(dates,res[k])
        axes[i].set_title('Historical distribution of asymmetric T parameter: {}'.format(k), fontsize=24, y=1.03)
        axes[i].tick_params(labelsize=16)
        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axes[i].set_xlabel('')
    figs={}   
    figs['res']=fig
    figs['pit']=figpit   
    figs['ls']=figls
    plt.close('all')
    return figs, chartpacks

def get_hist_kernel_dist(cond_quants, fitparam, dates, horizon):
    res = []
    kfits =[]
    h = fitparam['mode']['bandwidth']

    # perform fitting for  kernel model 
    for cond_quant in cond_quants:
        cond_quant_uncross = quantile_uncrossing(cond_quant)
        kfit = Weighted_kernel(cond_quant_uncross, bandwidth=h)
        kfit.w_kernel_fit()
        kfits.append(kfit)
    
    # select a fixed x
    x = select_kernel_x_list(kfits)

    for kfit, fitdate in zip(kfits, dates):
        res_fit, dfpdf = gen_kernel_values(fitdate, horizon, kfit, x)
        res_fit['dfpdf'] = dfpdf
        res.append(res_fit)
    return res

def get_model_quantiles(model, fittype):
    if fittype=='T-skew':
        v_q15=tskew_ppf(0.15, df=model['df'], loc=model['loc'], scale=model['scale'], skew=model['skew'])
        v_q40=tskew_ppf(0.4, df=model['df'], loc=model['loc'], scale=model['scale'], skew=model['skew'])
        v_q60=tskew_ppf(0.6, df=model['df'], loc=model['loc'], scale=model['scale'], skew=model['skew'])
        v_q85=tskew_ppf(0.85, df=model['df'], loc=model['loc'], scale=model['scale'], skew=model['skew'])
    elif fittype=='Asymmetric T':
        v_q15 = asymt_ppf(0.15, alpha=model['skew'], nu1=model['kleft'], nu2=model['kright'], mu=model['loc'], sigma=model['scale'])
        v_q40 = asymt_ppf(0.4, alpha=model['skew'], nu1=model['kleft'], nu2=model['kright'], mu=model['loc'], sigma=model['scale'])
        v_q60 = asymt_ppf(0.6, alpha=model['skew'], nu1=model['kleft'], nu2=model['kright'], mu=model['loc'], sigma=model['scale'])
        v_q85 = asymt_ppf(0.85, alpha=model['skew'], nu1=model['kleft'], nu2=model['kright'], mu=model['loc'], sigma=model['scale'])
    elif fittype=='Kernel-based':
        v_q15, v_q40, v_q60, v_q85 = model.w_kernel_ppf(np.array([0.15, 0.4, 0.6, 0.85]))
    return v_q15, v_q40, v_q60, v_q85

def select_x_list(model_fits, fittypes, modx):
     # extract the mode for each period
    loclist = np.array(modx)
    
    # set initial values
    min_v = min(loclist)-8
    max_v = max(loclist)+8
    
    for fittype, modelfit in zip(fittypes, model_fits):
        v_q15, v_q40, v_q60, v_q85 = get_model_quantiles(modelfit, fittype)
        
        # increase the range if some quantiles are outside
        min_v = min(min_v,v_q15-abs(v_q15-v_q40))
        max_v = max(max_v,v_q85+abs(v_q85-v_q60))
    x_list = np.array([x for x in np.linspace(min_v,max_v,500)])
    return x_list


    