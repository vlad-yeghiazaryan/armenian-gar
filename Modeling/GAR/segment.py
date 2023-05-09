## 3rd-party modules
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Internal modules
from .partition import retropolated_PCA
from .quantfit import condquant
from .tsfit import get_cond_quant, tskew_fit, asymt_fit
from .tsfit import gen_PDF_and_CDF, gen_asymt_PDF_and_CDF
from .tsfit import tskew_pdf
from .tsfit import tskew_cdf
from .tsfit import tskew_ppf
from .tsfit import tskew_mean
from .tsfit import asymt_pdf
from .tsfit import asymt_cdf
from .tsfit import asymt_ppf
from .tsfit import asymt_mean

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

# Functions for step 4: segment test
def run_segment(dict_input_segment, data, model=sm.QuantReg):
    '''
    Main run function for step 2, quantfit.

    Takes in as arguments a dict for input parameters
    and a df for data. Outputs a dict for output parameters.

    Does quantile fits and returns a dict of output parameters.
    ** This function should be independent of any Excel input/output
    and be executable as a regular Python function independent of Excel. **
    '''    
    # input definitions:
    # PCA inputs
    horizonlist = dict_input_segment['horizonlist']
    target =  dict_input_segment['target']
    method_growth = dict_input_segment['method_growth']
    retropolate = dict_input_segment['retropolate']
    dict_groups = dict_input_segment['partition_groups']
    transformer = dict_input_segment['transformer']
    df_partition = data.copy()

    # Q-fit inputs
    qlist = dict_input_segment['quantlist']
    qlist.sort()

    # T-skew fit inputs
    fitdate = dict_input_segment['fitdate']
    fitparam = dict_input_segment['fit_params']
    fitconstrainlist = dict_input_segment['fitconstrainlist']
    fitconstrainvalues = dict_input_segment['fitconstrainvalues']
    res = []
    
    # Run horizons
    for indh, horizon in enumerate(horizonlist):
        # Variable setup
        depvar  = target + '_hz_' + str(horizon)

        # PCA fit
        df_quantfit, partition_load, partition_log = retropolated_PCA(df_partition, dict_groups, target, horizon=horizon, method_growth=method_growth, retropolate=retropolate)

        # Data transformation
        df_quantfit = transformer.transform(df_quantfit, depvar)
        regressors = df_quantfit.drop(columns=['date', depvar]).columns
        
        # Q-fit
        df_quantcoef, dcond_quantiles_all, loco_all, exitcode = condquant(df_quantfit.set_index('date'), depvar, regressors, horizon, qlist, model)

        # Subset selection
        period = fitparam['qsmooth_period'] if type(fitparam['qsmooth_period'])!=str else horizon
        cond_quant = get_cond_quant(fitdate, df_quantfit, df_quantcoef, target, horizon, fitparam['qsmooth'], period)

        # Update the fitparam mode before every fit
        fitparam['mode']['constraint'] = fitconstrainlist[indh]
        fitparam['mode']['value'] = fitconstrainvalues[indh]

        # T-skew fit
        olsmean = cond_quant.pop('mean')
        if fitparam['fittype']=='T-skew':
            tsfit = tskew_fit(cond_quant, fitparam)

            # Generat PDF and CDF
            dfpdf = gen_PDF_and_CDF(tsfit)
        
         # Asymmetric T-skew fit
        elif fitparam['fittype']=='Asymmetric T':
            tsfit = asymt_fit(cond_quant, fitparam, olsmean)

            # Generat PDF and CDF
            dfpdf = gen_asymt_PDF_and_CDF(tsfit, tsfit['loc'])

        res_hz = {
            'horizon':horizon,
            'df_quantcoef':df_quantcoef,
            'olsmean':olsmean,
            'cond_quant': cond_quant,
            'tsfit': tsfit,
            'dfpdf': dfpdf,
            'loc': tsfit['loc'],
            # 'loc': tsfit['loc']/tsfit['scale'],
            }
        res.append(res_hz)

    # Plotting
    skewtlist = [res[i]['tsfit'] for i in range(len(horizonlist))]
    loclist = [res[i]['loc'] for i in range(len(horizonlist))]
    tails, fig, fig2, dfpdf = gen_seg_skewt([fitdate]*len(horizonlist), fitparam, skewtlist, horizonlist, loclist)

    # nhz = len(dict_input_segment['horizonlist'])
    # fig3 = coeff_plot(df_coefs, regressors, qlist, nhz)
    # hlist = list(hset).sort()
    # termfigs = termstruct_plot(df_term,regressors, qlist, hlist)
    segment_out = {
        'res':res,
        'tails':tails,
        'fig':fig,
        'fig2':fig2,
        'dfpdf':dfpdf
    }
    return segment_out

def gen_seg_skewt(fitdates,fitparam,skewtlist,horizonlist,loclist):
    n=len(skewtlist)
    colorlist=['red','blue','green','cyan','magenta','orange','lime','violet','crimson']
    ymax=-1
    if fitparam['fittype']=='T-skew':
        min_v = min(loclist)-8
        max_v = max(loclist)+8
        for indhz in range(n):
            tsfit=skewtlist[indhz]
            v_q5=tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            v_q40=tskew_ppf(0.4, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            v_q60=tskew_ppf(0.6, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            v_q95=tskew_ppf(0.95, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

            min_v = min(min_v,v_q5-abs(v_q5-v_q40))
            max_v = max(max_v,v_q95+abs(v_q95-v_q60))

        x_list = [x for x in np.arange(min_v,max_v,0.05)]
        titlestr = "T-skew forecast for growth rate"
        fig, ax = plt.subplots(1, 1, figsize=(20,10))
        ax.set_title(titlestr,fontsize=24)
        ax.legend(fontsize=24)
        ax.tick_params(labelsize=24)
        plt.legend(loc=2)
        plt.ylabel('Probability Density', fontsize=24)
        plt.xlabel('GDP (compound annual growth rate)', fontsize=24)    
        
        titlestr_cdf = "T-skew forecast for growth rate"
        fig2, ax2 = plt.subplots(1, 1, figsize=(20,10))
        ax2.set_title(titlestr_cdf,fontsize=24)
        ax2.set_ylim(0, 1)
        ax2.set_title(titlestr,fontsize=24)        
        ax2.legend(fontsize=24,loc=2)
        ax2.tick_params(labelsize=24)
        plt.legend(loc=2)
        plt.xlim(x_list[0],x_list[-1])
        plt.ylabel('Cumulative probability', fontsize=24)
        plt.xlabel( 'GDP (compound annual growth rate)', fontsize=24)
        df_header=['Tskew_PDF_x']
    elif fitparam['fittype']=='Asymmetric T':
        min_v = min(loclist)-1.5
        max_v = max(loclist)+1.5
        x_list = [x for x in np.arange(min_v,max_v,0.01)]
        titlestr = "Asymmetric T forecast for growth rate"
        fig, ax = plt.subplots(1, 1, figsize=(20,10))
        ax.set_title(titlestr,fontsize=24)
        ax.legend(fontsize=24)
        ax.tick_params(labelsize=24)
        plt.legend(loc=2)
        plt.ylabel('Probability Density', fontsize=24)
        plt.xlabel('GDP (compound annual growth rate)', fontsize=24)    
        
        titlestr_cdf = "Asymmetric T forecast for growth rate"
        fig2, ax2 = plt.subplots(1, 1, figsize=(20,10))
        ax2.set_title(titlestr_cdf,fontsize=24)
        ax2.set_ylim(0, 1)
        ax2.set_title(titlestr,fontsize=24)        
        ax2.legend(fontsize=24,loc=2)
        ax2.tick_params(labelsize=24)
        plt.legend(loc=2)
        plt.xlim(x_list[0],x_list[-1])
        plt.ylabel('Cumulative probability', fontsize=24)
        plt.xlabel('GDP (compound annual growth rate)', fontsize=24)
        df_header=['AsymT_PDF_x']
    
    
    df_tmp=[x_list]
    horizons=['Forward horizon']
    horizons.extend(horizonlist)
    cmode=['Conditional mode']
    cmedian=['Conditional median']
    cmean=['Conditional mean']
    gar5=['GaR5%']
    gar10=['GaR10%']
    gzero=['Growth below 0 probablity']
    xq5s=[]
    for indhz in range(n):
        
        if fitparam['fittype']=='T-skew':
            tsfit=skewtlist[indhz]
            yvals= [tskew_pdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]    
            ymax=max(ymax,max(yvals))
            ycdf = [tskew_cdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]
            yzero=tskew_cdf(0, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            ax.plot(x_list,yvals,'-',color=colorlist[indhz],label=fitdates[indhz].strftime('%m/%d/%Y')+" forward "+str(horizonlist[indhz]))
            ax2.plot(x_list,ycdf,'-',color=colorlist[indhz],label=fitdates[indhz].strftime('%m/%d/%Y')+" forward "+str(horizonlist[indhz]))
            df_header.append('Tskew_PDF_y_PROJ'+str(indhz+1))
            df_header.append('Tskew_CDF_y_PROJ'+str(indhz+1))
            df_tmp.append(yvals)
            df_tmp.append(ycdf)
        
            xq5=tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
            xq10=tskew_ppf(0.1, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
            yq5= tskew_pdf(xq5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
            yq10= tskew_pdf(xq10, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])     
            ycq5= tskew_cdf(xq5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) 
            ycq10= tskew_cdf(xq10, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            meanx=tskew_mean(df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            modx=tsfit['loc']
            medx=tskew_ppf(0.5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
            xq5s.append(xq10)
    
        elif fitparam['fittype']=='Asymmetric T':
            asymtfit=skewtlist[indhz]
            yvals= [asymt_pdf(z, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) for z in x_list]    
            ymax=max(ymax,1.2*max(yvals))
            ycdf = [asymt_cdf(z, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) for z in x_list]
            
            ax.plot(x_list,yvals,'-',color=colorlist[indhz],label=fitdates[indhz].strftime('%m/%d/%Y')+" forward "+str(horizonlist[indhz]))
            ax2.plot(x_list,ycdf,'-',color=colorlist[indhz],label=fitdates[indhz].strftime('%m/%d/%Y')+" forward "+str(horizonlist[indhz]))
            df_header.append('AsymT_PDF_y_PROJ'+str(indhz+1))
            df_header.append('AsymT_CDF_y_PROJ'+str(indhz+1))
            df_tmp.append(yvals)
            df_tmp.append(ycdf)
            xq5 = asymt_ppf(0.05, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
            xq10 = asymt_ppf(0.1, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])     
            yq5= asymt_pdf(xq5, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
            yq10= asymt_pdf(xq10,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
            ycq5= asymt_cdf(xq5, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
            ycq10= asymt_cdf(xq10,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
            yzero=asymt_cdf(0,alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale']) 
            meanx=asymt_mean(alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
            modx=loclist[indhz]
            medx=asymt_ppf(0.5, alpha=asymtfit['skew'], nu1=asymtfit['kleft'], nu2=asymtfit['kright'], mu=asymtfit['loc'], sigma=asymtfit['scale'])
            xq5s.append(xq5)
            
        cmode.append(float("{:.4f}".format(modx)))
        cmedian.append(float("{:.4f}".format(medx)))
        cmean.append(float("{:.4f}".format(meanx)))
        gar5.append(float("{:.4f}".format(xq5)))
        gar10.append(float("{:.4f}".format(xq10)))
        gzero.append(float("{:.4f}".format(yzero)))
            
            
    dfpdf=pd.DataFrame(df_tmp)
    dfpdf = dfpdf.transpose()
    dfpdf.columns = df_header
    if fitparam['fittype']=='T-skew':
        ax.set_ylim(0, 1.2*ymax)
    elif fitparam['fittype']=='Asymmetric T':
        ax.set_ylim(0,1.05*ymax)
        c=(min(loclist)+max(loclist))/2
        l=max(xq5s)
        ax.set_xlim(l,1.8*c-0.8*l)
        ax2.set_xlim(l,1.8*c-0.8*l)
    ax.legend(fontsize=24,loc=2)
    ax2.legend(fontsize=24,loc=2)
    res=[horizons,cmode,cmedian,cmean,gar5,gar10,gzero]
    plt.close('all')
    return res, fig, fig2, dfpdf 

# Coefficients plotting
def coeff_plot(dcoeffc, regressors, qlist, nhz):   
    qlist.sort()
    
    for i in range(len(qlist)):
        if qlist[i]==0.5:
            ind05=i
            break
    qlist.insert(ind05,'mean')

    ## Variables text
    variable_list_coeff = list(regressors)
    variable_list_coeff.sort()
    n=len(variable_list_coeff)

    # Style of the charts
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(n, 1, figsize=(20,9*n))

    ## Plots    
    colorlist=['red','blue','green','cyan','magenta','orange','lime','violet','crimson']
    inds=np.arange(len(qlist))
    bar_width=0.1
    for v, variable in enumerate(variable_list_coeff):
        vs=variable.split('_trans_')
        varn=vs[0]
        if vs[1][-4:]!='None':
            varn+='_'+vs[1]
        if len(varn)>20:
            variable_label = varn[:17]+'...'
        else:
            variable_label = varn
        maxv=-99999999                    
        for hind in range(nhz):
            cn=[]
            for q in qlist:
                cn.append(dcoeffc[(dcoeffc.index==variable) & (dcoeffc['quantile']==q)]['coeff_scale_PROJ'+str(hind+1)].values[0])
            maxv=max(maxv,max([abs(a) for a in cn]))
            if n>1:
                ax[v].bar(inds+hind*bar_width,cn,bar_width,alpha=0.7,color=colorlist[hind],label='PROJ'+str(hind+1))
                plt.sca(ax[v])
            else:
                ax.bar(inds+hind*bar_width,cn,bar_width,alpha=0.7,color=colorlist[hind],label='PROJ'+str(hind+1))
            plt.xticks(inds+nhz//2*bar_width,qlist)
        '''    
        dcv = dcoeffc.loc[(dcoeffc.variable == variable),:].copy()
        dcv = dcv.reset_index()
        dcv = dcv.set_index(dcv['quantile'])
        dcv = dcv.reindex(qlist)
        erna=dcv['errors'].isnull().any()
            # Plot the coefficients
        if erna:
            dcv['coeff_scale'].plot.bar(color='blue',ax=axes[v])
            x=max(abs(min(dcv['coeff_scale'].values)),abs(max(dcv['coeff_scale'].values)))
        else:
            dcv['coeff_scale'].plot.bar(color='blue',yerr = dcv.errors,ax=axes[v])
            x=max(abs(min(dcv['lower'].values)),abs(max(dcv['upper'].values)))
        '''
        if n>1:
            ax[v].axhline(y=0, c='black', linewidth=0.7)
            ax[v].set_title('{0}'.format(variable_label), fontsize=25, y=1.05)
            ax[v].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[v].set_xlabel('')
            ax[v].tick_params(labelsize=25)  
            ax[v].legend(fontsize=20,bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
            ax[v].set_ylim(-1.1*maxv,1.1*maxv)
        else:
            ax.axhline(y=0, c='black', linewidth=0.7)
            ax.set_title('{0}'.format(variable_label), fontsize=25, y=1.05)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_xlabel('')
            ax.tick_params(labelsize=25)  
            ax.legend(fontsize=20,bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
            ax.set_ylim(-1.1*maxv,1.1*maxv)
    if n>1:   
        fig.suptitle('Quantile regressions coefficients',y=0.92,fontsize=30)
    else:
        fig.suptitle('Quantile regressions coefficients',y=1,fontsize=30)
    plt.close('all') 
    return(fig)

# Coefficients plotting
def termstruct_plot(df_term,regressors, qlist, hlist):
    ## Variables text
    variable_list_coeff = list(regressors)
    variable_list_coeff.sort()
    
    ## Define the grid
    n=len(variable_list_coeff)
    m=len(hlist)
    if m<=4:
        cs=m
    elif m<=6:
        cs=3
    else:
        cs=4
    rs=m//4+1
    termfigs=[]

    # Style of the charts
    plt.style.use('seaborn-white')
    
    # Plotting
    for v, variable in enumerate(variable_list_coeff):  
        fig = plt.figure(figsize=(cs*8,8*rs+2))
        axes=[]
        gs = GridSpec((n+1)//4+1, min(4,m),hspace=0.35)
        
        # Plots    
        vs=variable.split('_trans_')
        varn=vs[0]
        if vs[1][-4:]!='None':
            varn+='_'+vs[1]
        for i in range(m):
            axes.append(fig.add_subplot(gs[i//4,i%4]))
                        
            dcv = df_term.loc[(df_term.index == variable),:].copy()
            dcv = dcv.reset_index()
            dcv = dcv.set_index(dcv['quantile'])
            dcv = dcv.reindex(qlist)
            coff='coeff_scale_hz'+str(hlist[i])
            erro='error_hz'+str(hlist[i])
            upper='upper_hz'+str(hlist[i])
            lower='lower_hz'+str(hlist[i])
            erna=dcv[erro].isnull().any()
            # Plot the coefficients
            if erna:
                dcv[erro].plot.bar(color='blue',ax=axes[i])
                x=max(abs(min(dcv[coff].values)),abs(max(dcv[coff].values)))
            else:
                dcv[coff].plot.bar(color='blue',yerr = dcv[erro],ax=axes[i])    
                
                x=max(abs(min(dcv[lower].values)),abs(max(dcv[upper].values)))
                
            axes[i].axhline(y=0, c='black', linewidth=0.7)
            axes[i].set_title('{0}'.format('Horizon '+str(hlist[i])), fontsize=25, y=1.02)
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axes[i].set_xlabel('')
            
            axes[i].set_ylim(-x-0.1,x+0.1)
            axes[i].tick_params(labelsize=25)
        
        fig.suptitle('Term structure for '+varn, y=1,fontsize=30)
        termfigs.append(fig)

    plt.close('all')
    return termfigs
