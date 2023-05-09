## 3rd-party modules
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import t, norm
from scipy import interpolate
from scipy.special import gamma
from scipy.optimize import minimize
import math 
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import matplotlib.pyplot as plt  
from matplotlib.ticker import FormatStrFormatter

def run_tsfit(fitdate, fitparam, data, qcoef, target, horizon):
    '''
    Main run function for step 3, tsfit.

    Takes in as arguments a dict for input parameters
    and a df for data. Outputs a dict for output parameters.

    Does quantile fits and returns a dict of output parameters.
    ** This function should be independent of any Excel input/output
    and be executable as a regular Python function independent of Excel. **
    '''
    # ------------------------
    # Create output dict
    # ------------------------
    dict_output_tsfit = dict()

    # Param setup
    cond_quant = get_cond_quant(fitdate, data, qcoef, target, horizon, fitparam['qsmooth'], fitparam['qsmooth_period'])

    # Estimation
    res, cq, fig, fig2, dfpdf = gen_skewt(fitdate, fitparam, cond_quant, horizon)
    if fitparam['fittype']=='Asymmetric T':
        dfpdf=dfpdf[['AsymT_PDF_x','AsymT_CDF','AsymT_PDF_y']].round(decimals=5)
    elif fitparam['fittype']=='T-skew':
        dfpdf=dfpdf[['Tskew_PDF_x','Tskew_CDF','Tskew_PDF_y']].round(decimals=5)
    dict_output_tsfit['result'] = res
    dict_output_tsfit['data']   = cq
    dict_output_tsfit['fig']    = fig
    dict_output_tsfit['fig2']    = fig2
    dict_output_tsfit['dfpdf']    = dfpdf
    return dict_output_tsfit

def select_df_partition(fitdate, df, target, horizon, qsmooth='None', qsmooth_per=2):
    # Fitdat
    depvar  = target + '_hz_' + str(horizon)
    df = df.copy()
    df.set_index('date', inplace=True)
    if qsmooth=='None':
        try:
            df_partition_fit = df.loc[fitdate]
        except:
            df_partition_fit = df.iloc[-1,:]
            print('The latest date in the data will be used.')
                           
    elif qsmooth=='Median':
        per = int(qsmooth_per)
        df_partition_fit = df[df.index<=fitdate].tail(per).median()
        
    elif qsmooth=='Mean':        
        per=int(qsmooth_per)
        df_partition_fit = df[df.index<=fitdate].tail(per).mean()
    
    df_partition_fit.drop(['date', depvar], inplace=True, errors='ignore')
    df_partition_fit['const'] = 1
    df_partition_fit = df_partition_fit.sort_index()
    return df_partition_fit

def get_cond_quant(fitdate, data, qcoef, target, horizon, qsmooth='None', qsmooth_per=2):
    df_partition_fit = select_df_partition(fitdate, data, target, horizon, qsmooth, qsmooth_per)
    cond_quant = qcoef.groupby('quantile').apply(lambda x: df_partition_fit @ x.set_index('variable')['coeff_noscale'].sort_index())
    return cond_quant.to_dict()

def gen_skewt(fitdate, fitparam, cond_quant, horizon):
    # Add return values
    olsmean = cond_quant.pop('mean')
    if fitparam['qsmooth']=='None':
        cqlist=['Tau','Cond_quant']
    else:
        cqlist=['Tau','Cond_quant_smoothed']     
    cq = pd.DataFrame(cond_quant.items(), columns=cqlist)

    if fitparam['fittype']=='T-skew':
        # fitting the T-skew to data
        tsfit = tskew_fit(cond_quant,fitparam)

        # generating data for PDF and CDF
        dfpdf = gen_PDF_and_CDF(tsfit)

        # calc some variables
        xq5=tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
        xq10=tskew_ppf(0.1, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        medx=tskew_ppf(0.5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        meanx=tskew_mean(df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

        # Selecting the loc for the tskew fit
        if fitparam['mode']['constraint']=='Fixed':
            loc = fitparam['mode']['value']
        elif fitparam['mode']['constraint']=='Free':
            loc = tsfit['loc'] # is this correct !!!?
            # loc=tsfit['loc']/tsfit['scale'] 
        else:
            loc = cond_quant[0.5]

        # plotting
        fig, fig2 = plot_T_dist(fitdate, dfpdf, fitparam, tsfit, loc, horizon)

        res = {
            'Date of input': fitdate,
            'Horizon forward': horizon,
            'Conditional mode': loc,
            'Conditional median': medx,
            'Conditional mean': meanx,
            'GaR5%': xq5,
            'GaR10%': xq10,
            'Scale': tsfit['scale'],
            'Skewness': tsfit['skew']
        }
        return res, cq, fig, fig2, dfpdf
    
    elif fitparam['fittype']=='Asymmetric T':
        # fitting the asymt T-skew to data
        asymtfit = asymt_fit(cond_quant, fitparam, olsmean)

        # Selecting the loc for the tskew fit
        if fitparam['mode']['constraint']=='Fixed':
            loc = fitparam['mode']['value']
        elif fitparam['mode']['constraint']=='Free':
            loc=asymtfit['loc']
        else:
            loc = cond_quant[0.5]
        
        # generating data for PDF and CDF
        dfpdf = gen_asymt_PDF_and_CDF(asymtfit, loc)
        
        # calc some variables
        medx=asymt_ppf(0.5, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
        meanx=asymt_mean(alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
        xq5 = asymt_ppf(0.05, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
        xq10 = asymt_ppf(0.1, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])

        # plotting
        fig, fig2 = plot_asymt_T_dist(fitdate, dfpdf, fitparam, asymtfit, loc, horizon)

        res = {
            'Date of input': fitdate,
            'Horizon forward': horizon,
            'Conditional mode': loc,
            'Conditional median': medx,
            'Conditional mean': meanx,
            'GaR5%': xq5,
            'GaR10%': xq10,
            'Scale': asymtfit['scale'],
            'Left kurtosis': asymtfit['kleft'],
            'Right kurtosis': asymtfit['kright'],
            'Skewness': asymtfit['skew']
        }
        return res, cq, fig, fig2, dfpdf

def gen_PDF_and_CDF(tsfit, x_list=None):
    v_q5 = tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    v_q40 = tskew_ppf(0.4, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    v_q60 = tskew_ppf(0.6, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    v_q95 = tskew_ppf(0.95, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    min_v = v_q5-abs(v_q5-v_q40)
    max_v = v_q95+abs(v_q95-v_q60)
    while tskew_cdf(min_v+1, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])>0.05:
        min_v-=1
    
    if type(x_list)==type(None):
        x_list = [x for x in np.arange(min_v,max_v,0.05)]
    yvals = [tskew_pdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]    
    ycdf = [tskew_cdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]
    tmp_dic={'Tskew_PDF_x':x_list,'Tskew_PDF_y':yvals,'Tskew_CDF':ycdf}
    dfpdf = pd.DataFrame(tmp_dic)
    return dfpdf

def tskew_ppf_vec(quantiles, tsfit):
    ppf = {q:tskew_ppf(q, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for q in quantiles}
    return ppf

def asymt_ppf_vec(quantiles, asymtfit):
    ppf = {q: asymt_ppf(q, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) for q in quantiles}
    return ppf

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

def plot_T_dist(fitdate, dfpdf, fitparam, tsfit, loc, horizon):
    x_list, yvals, ycdf = dfpdf.values.T
    v_q5 = tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    v_q40 = tskew_ppf(0.4, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    x_inc = (v_q40-v_q5)/4

    for i,y in enumerate(ycdf):
        q5loc=i
        if y>0.05:               
            break
    
    xq5=tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
    xq10=tskew_ppf(0.1, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
    yq5= tskew_pdf(xq5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
    yq10= tskew_pdf(xq10, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])     
    ycq5= tskew_cdf(xq5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
    ycq10= tskew_cdf(xq10, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    
    meanx=tskew_mean(df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    titlestr = "T-skew quantile fit for "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    lablestr = "Density "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.set_ylim(0, 1.2 * max(yvals))
    ax.set_title(titlestr,fontsize=24)
    ax.fill_between(x_list[:q5loc], 0, yvals[:q5loc],  facecolor='red', interpolate=True)
    ax.plot(x_list,yvals,'b-',label=lablestr)

    modx=tsfit['loc']
    mody=tskew_pdf(loc, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    
    medx=tskew_ppf(0.5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    medy=tskew_pdf(medx, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    meany=tskew_pdf(meanx, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    
    if fitparam['plot_mode']:
        ax.plot([modx,modx],[0,mody],'r-.')
        ax.annotate('Mode', xy=(modx, mody),xycoords='data',
                    xytext=(modx+x_inc, mody*1.2), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),fontsize=24,)
    if fitparam['plot_median']:
        ax.plot([medx,medx],[0,medy],'m-.')
        if medx<modx:
            sp=-x_inc
        else:
            sp=x_inc
        ax.annotate('Median', xy=(medx, medy),xycoords='data',
                    xytext=(medx+sp, mody*1.1), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),fontsize=24,)
    if fitparam['plot_mean']:
        if meanx<modx:
            sp=-x_inc
        else:
            sp=x_inc
        ax.plot([meanx,meanx],[0,meany],'c-.')
        ax.annotate('Mean', xy=(meanx, meany),xycoords='data',
                    xytext=(meanx+sp, meany*1.1), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),fontsize=24,)
        
    ax.plot([xq5,xq5],[0,yq5],'k--')
    
    ax.annotate('GaR 5%', xy=(xq5, yq5), xycoords='data',
                xytext=(xq5-x_inc, yq5*1.4), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=90,angleB=0"),fontsize=24,)
    ax.plot([xq10,xq10],[0,yq10],'k--')
    ax.annotate('GaR 10%', xy=(xq10, yq10), xycoords='data',
                xytext=(xq10-x_inc, yq10*1.3), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                connectionstyle="angle3,angleA=0,angleB=90"),fontsize=24,)
    ax.legend(fontsize=24)
    ax.tick_params(labelsize=24)
    plt.ylim(0, max(yvals)*1.4)
    plt.xlim(x_list[0],x_list[-1])
    
    plt.ylabel('Probability Density', fontsize=24)
    plt.xlabel('GDP (compound annual growth rate)', fontsize=24)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(20,10))
    titlestr = "T-skew quantile fit for "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    lablestr = "Cumulative probability "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    ax2.fill_between(x_list[:q5loc+1], 0, ycdf[:q5loc+1],  facecolor='red', interpolate=True)
    ax2.plot([xq5,xq5],[0,ycq5],'k--')
    ax2.plot([xq10,xq10],[0,ycq10],'k--')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    modcy=tskew_cdf(modx, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    medcy=tskew_cdf(medx, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    meancy=tskew_cdf(meanx, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
    
    if fitparam['plot_mode']:
        ax2.plot([modx,modx],[0,modcy],'r-.')
        ax2.plot([x_list[0],modx],[modcy,modcy],'r-.')
        
    if fitparam['plot_median']:
        ax2.plot([medx,medx],[0,medcy],'m-.')
        ax2.plot([x_list[0],medx],[medcy,medcy],'m-.')
    if fitparam['plot_mean']:
        ax2.plot([meanx,meanx],[0,meancy],'c-.')
        ax2.plot([x_list[0],meanx],[meancy,meancy],'c-.')
    
    ax2.set_ylim(0, 1)
    ax2.set_title(titlestr,fontsize=24)        
    ax2.plot(x_list,ycdf,'b-',label=lablestr)
    ax2.legend(fontsize=24,loc=2)
    ax2.tick_params(labelsize=24)
    plt.ylim(0, 1)
    plt.xlim(x_list[0],x_list[-1])
    plt.ylabel('Cumulative Probability', fontsize=24)
    plt.xlabel('GDP(compound annual growth rate)', fontsize=24)
    plt.close('all')
    return fig, fig2

def plot_asymt_T_dist(fitdate, dfpdf, fitparam, asymtfit, loc, horizon):
    x_list, yvals, ycdf = dfpdf.values.T
    meanx=asymt_mean(alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])

    for i,y in enumerate(ycdf):
        if y>0.05:
            q5loc=i
            break
    for i,y in enumerate(ycdf):
        if y>0.1:
            q10loc=i
            break

    xq5 = asymt_ppf(0.05, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    xq10 = asymt_ppf(0.1, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])     
    yq5= asymt_pdf(xq5, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    yq10= asymt_pdf(xq10,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    ycq5= asymt_cdf(xq5, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    ycq10= asymt_cdf(xq10,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 

    
    titlestr = "Asymmetric T quantile fit for "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    lablestr = "Density "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.set_ylim(0, 1.2 * max(yvals))
    ax.set_title(titlestr,fontsize=24)
    ax.fill_between(x_list[:q5loc+1], 0, yvals[:q5loc+1],  facecolor='red', interpolate=True)
    ax.plot(x_list,yvals,'b-',label=lablestr)
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    modx=loc
    mody=asymt_pdf(loc,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    
    medx=asymt_ppf(0.5, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    medy=asymt_pdf(medx, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])        
    meany=asymt_pdf(meanx, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
    
    if fitparam['plot_mode']:
        ax.plot([modx,modx],[0,mody],'r-.')
        ax.annotate('Mode', xy=(modx, mody),xycoords='data',
                    xytext=(min(modx*1.05,modx+0.2), mody*1.2), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),fontsize=24,)
        
    if fitparam['plot_median']:
        if medx<modx:
            sp=-1
        else:
            sp=1
        ax.plot([medx,medx],[0,medy],'m-.')
        ax.annotate('Median', xy=(medx, medy),xycoords='data',
                    xytext=(min(medx*(1+0.05*sp),medx+0.2*sp), medy*1.1), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),fontsize=24,)
    if fitparam['plot_mean']:
        if meanx<modx:
            sp=-1
        else:
            sp=1
        ax.plot([meanx,meanx],[0,meany],'c-.')
        ax.annotate('Mean', xy=(meanx, meany),xycoords='data',
                    xytext=(min(meanx*(1+0.05*sp),meanx+0.2*sp), meany*1.1), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),fontsize=24,)
        
    ax.plot([xq5,xq5],[0,yq5],'k--')

    ax.annotate('GaR 5%', xy=(x_list[q5loc-1], yq5), xycoords='data',
                xytext=(x_list[q5loc-1]*0.9, (yq5+mody)/3), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=90,angleB=0"),fontsize=24,)
    ax.plot([xq10,xq10],[0,yq10],'k--')
    ax.annotate('GaR 10%', xy=(x_list[q10loc-1], yq10), xycoords='data',
                xytext=(x_list[q10loc-1]*0.9, (yq10+mody)/2), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                connectionstyle="angle3,angleA=0,angleB=90"),fontsize=24,)
    ax.legend(fontsize=24)
    ax.tick_params(labelsize=24)
    plt.ylim(0, max(yvals)*1.4)
    plt.xlim(x_list[0],x_list[-1])
    plt.ylabel('Probability Density', fontsize=24)
    plt.xlabel('GDP (compound annual growth rate)', fontsize=24)

    fig2, ax2 = plt.subplots(1, 1, figsize=(20,10))
    titlestr = "Asymmetric T quantile fit for "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    lablestr = "Cumulative probability "+fitdate.strftime('%m/%d/%Y')+" "+"growth rate"+" forward "+str(horizon)
    
    ax2.set_ylim(0, 1)
    ax2.set_title(titlestr,fontsize=24)        
    ax2.plot(x_list,ycdf,'b-',label=lablestr)
    ax2.fill_between(x_list[:q5loc], 0, ycdf[:q5loc],  facecolor='red', interpolate=True)
    ax2.plot([xq5,xq5],[0,ycq5],'k--')
    ax2.plot([xq10,xq10],[0,ycq10],'k--')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    modcy=asymt_cdf(modx,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    medcy=asymt_cdf(medx,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    meancy=asymt_cdf(meanx,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
    if fitparam['plot_mode']:
        ax2.plot([modx,modx],[0,modcy],'r-.')
        ax2.plot([x_list[0],modx],[modcy,modcy],'r-.')
        
    if fitparam['plot_median']:
        ax2.plot([medx,medx],[0,medcy],'m-.')
        ax2.plot([x_list[0],medx],[medcy,medcy],'m-.')
    if fitparam['plot_mean']:

        ax2.plot([meanx,meanx],[0,meancy],'c-.')
        ax2.plot([x_list[0],meanx],[meancy,meancy],'c-.')
    ax2.legend(fontsize=24,loc=2)
    ax2.tick_params(labelsize=24)
    plt.ylim(0,1)
    plt.xlim(x_list[0],x_list[-1])
    plt.ylabel('Cumulative Probability', fontsize=24)
    plt.xlabel('GDP (compound annual growth rate)', fontsize=24)
    plt.close('all')
    return fig, fig2

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

# get mean value of t-skew
def tskew_mean(df, loc, scale, skew):
    """
    Note by C. Wang
    Formula from Equation 5 of
    On Bayesian Modeling of Fat Tails and Skewness
    Carmen Fernandez and Mark F. J. Stee, 1998 JASA
    """
    cons1=skew-1/skew
    Mr=math.gamma((df+1)/2)/(math.sqrt(df*math.pi)*math.gamma(df/2))*2*df*scale
    return cons1*Mr+loc

# Ancillary functions, cf. Zhu and Galbraith JoE 2010
def K_plain(nu):
    top = gamma((nu+1)/2)
    bottom = np.sqrt(sp.pi*nu)*gamma(nu/2)
    return(top/bottom)

def alpha_star_plain(alpha, nu1, nu2):
    top = alpha*K(nu1)
    bottom = alpha*K(nu1) + (1-alpha)*K(nu2)
    return(top/bottom)

## To improve speed, vectorize the ancillary functions (used everywhere else)
K = np.vectorize(K_plain, otypes=[np.float], cache=False)
alpha_star = np.vectorize(alpha_star_plain, otypes=[np.float], cache=False)

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

# get mean value of t-skew
def tskew_mean(df, loc, scale, skew):
    """
    Note by C. Wang
    Formula from Equation 5 of
    On Bayesian Modeling of Fat Tails and Skewness
    Carmen Fernandez and Mark F. J. Stee, 1998 JASA
    """
    cons1=skew-1/skew
    Mr=math.gamma((df+1)/2)/(math.sqrt(df*math.pi)*math.gamma(df/2))*2*df*scale
    return cons1*Mr+loc

# Risk analysis : Delta VaR under different scenarios
def delta_VaR(quant_model, central_scenario, simulated_scenario):
    """ 
    Compute the variation of the VaR from a central scenario 

    Inputs:
    - quant_model: a quantile_reg object, including the regression coefficients
    - central scenario: values of the covariates to estimate the model on
    - simulated scenario: another set of covariates to measure the delta risk
    
    Output:
    - the parameters of the tskew-fit under both scenarios

    """

    ## Shorter names
    qr = quant_model
    
    def opt_parameters(scenario):
        """ Return the optimal parameters associated with a scenario """
        
        ## Compute the associated simulated quantiles
        sq = qr.cond_quantiles(predictors=scenario) 

        ## Retrieve the conditional quantiles, conditional mean and df
        quantile_list = qr.quantile_list
        cond_quant = sq.loc[sq['tau'] != 'mean','conditional_quantile_mean']
        cond_mean = sq.loc[sq['tau'] == 'mean','conditional_quantile_mean'].values[0]
        cq_variables = ['conditional_quantile_mean',
                        'conditional_quantile_mean_ci_lower',
                        'conditional_quantile_mean_ci_upper']
        cond_var = np.nanvar(sq.loc[:, cq_variables].values)
        degree_freedom = qr.data.shape[0] - len(qr.regressors) - 1

        ## Estimate the tskew_fit associated with the conditional quantiles
        tsk = tskew_fit(quantile_list=quantile_list,
                        conditional_quantiles=cond_quant,
                        df=degree_freedom, loc=cond_mean, cond_var=cond_var)

        return(tsk)


    ## Compute the optimal parameters for both scenarios
    c_tsk = opt_parameters(central_scenario) # Central
    s_tsk = opt_parameters(simulated_scenario) ## Simulated

    ## Package them into a frame
    dc = pd.DataFrame.from_dict(c_tsk, orient='index').transpose()
    dc.index = ['central']
    dc.insert(0,'scenario', 'central')

    ds = pd.DataFrame.from_dict(s_tsk, orient='index').transpose()
    ds.index = ['simulated']
    ds.insert(0,'scenario', 'simulated')

    dfinal = pd.concat([dc, ds], axis=0)
    
    return(dfinal)

# sampling t-skew via inverse transform sampling
def tskew_sampling(n,tau, df, loc, scale, skew):
    uni=np.random.random_sample(size=n)
    samples=[tskew_ppf(x,tau, df, loc, scale, skew) for x in uni]
    return samples
