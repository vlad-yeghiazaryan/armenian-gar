## 3rd-party modules
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Internal modules
from .growth import calc_growth_rate
from .partition import retropolated_PCA
from .quantfit import condquant
from .tsfit import get_cond_quant, tskew_fit, asymt_fit
from .tsfit import gen_PDF_and_CDF
from .tsfit import quantile_uncrossing, Weighted_kernel
from .historical import select_x_list

# Functions for step 4: segment test
def run_segment(dict_input_segment, data, model=sm.QuantReg):
    # input definitions:
    # PCA inputs
    horizonlist = dict_input_segment['horizonlist']
    base_target =  dict_input_segment['base_target']
    method_growth = dict_input_segment['method_growth']
    method = dict_input_segment['method']
    benchcutoff = dict_input_segment['benchcutoff']
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
        # PCA fit
        target = base_target+'_hz_'+str(horizon)
        df_partition[target] = calc_growth_rate(df_partition[base_target], horizon=horizon, yearfreq=4, method_growth=method_growth)
       
        df_quantfit, partition_load = retropolated_PCA(df_partition, dict_groups, target, horizon, method_growth, method, benchcutoff)

        # Data transformation
        if type(transformer)!=type(None):
            df_quantfit = transformer.transform(df_quantfit, target)
        
        # Q-fit
        regressors = df_quantfit.drop(columns=['date', target]).columns
        df_quantcoef, dcond_quantiles_all, loco_all, exitcode = condquant(df_quantfit.set_index('date'), target, regressors, horizon, qlist, model)

        # Subset selection
        period = fitparam['qsmooth_period'] if type(fitparam['qsmooth_period'])!=str else horizon
        cond_quant = get_cond_quant(fitdate, df_quantfit, df_quantcoef, target, horizon, fitparam['qsmooth'], period)

        # Update the fitparam mode before every fit
        fitparam['mode']['constraint'] = fitconstrainlist[indh]
        fitparam['mode']['value'] = fitconstrainvalues[indh]

        # T-skew fit
        olsmean = cond_quant.pop('mean')
        if fitparam['fittype']=='T-skew':
            model_fit = tskew_fit(cond_quant, fitparam)
            modx = model_fit['loc']
        
         # Asymmetric T-skew fit
        elif fitparam['fittype']=='Asymmetric T':
            model_fit = asymt_fit(cond_quant, fitparam, olsmean)
            modx = model_fit['loc']
        
        # Kernel fit
        elif fitparam['fittype']=='Kernel-based':
            h = fitparam['mode']['bandwidth']
            cond_quant_uncross = quantile_uncrossing(cond_quant)
            model_fit = Weighted_kernel(cond_quant_uncross, bandwidth=h)
            model_fit.w_kernel_fit()

            # getting the mode
            x = model_fit.q_values
            ypdf = model_fit.w_kernel_pdf(x)
            modx = x[np.argmax(ypdf)]
            
        res_hz = {
            'horizon':horizon,
            'df_quantcoef':df_quantcoef,
            'olsmean':olsmean,
            'cond_quant': cond_quant,
            'model_fit': model_fit,
            'modx': modx,
            }
        res.append(res_hz)

    model_fits = [res_fit['model_fit'] for res_fit in res]
    x = select_x_list(model_fits, fitparam['fittype'])
    
    # get dfpdf using a fixed set for x
    for res_fit in (res):
        model_fit = res_fit['model_fit']
        loc = res_fit['modx']
        res_fit['dfpdf'] = gen_PDF_and_CDF(model_fit, fitparam, x, loc)

    res = pd.DataFrame(res)
    return res
