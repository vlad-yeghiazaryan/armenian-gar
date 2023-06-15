
from datetime import datetime as date

## 3rd-party modules
import pandas as pd
from sklearn.preprocessing import scale
import statsmodels.api as sm
from statsmodels.api import QuantReg

# Functions for step 2: quantfit
def run_quantfit(data, target, horizon=4, model=QuantReg,
                 quantlist=[0.1, 0.25, 0.5, 0.75, 0.9]):
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
    dict_output_quantfit = dict()
    
    # ------------------------
    # Get parameters from
    # dict_input_quantfit
    # ------------------------
    depvar  = target + '_hz_' + str(horizon)
    df_quantfit = data.copy()
    regressors = df_quantfit.drop(columns=['date', depvar]).columns
    df_quantfit.set_index('date', inplace=True)

    # ------------------------
    # Run the quantfit
    # ------------------------
    qcoeff, cond_quant, local_prj, exitcode = condquant(df_quantfit, depvar, regressors, horizon, quantlist, model)

    # Add return values
    dict_output_quantfit['qcoef']      = qcoeff
    dict_output_quantfit['cond_quant'] = cond_quant
    dict_output_quantfit['localprj']    = local_prj
    
    return dict_output_quantfit

def condquant(dall, depvar,regressors_avl, horizon, ql, model):
#if 1==1:    
    ql.sort()
    c_id_dict = {'horizon' : horizon}
    
    dall=dall.dropna(subset=regressors_avl)
    qrs = QuantileReg(depvar, indvars=regressors_avl,
                      quantile_list=ql,
                      data=dall,
                      scaling=True, alpha=0.1)

    dc = qrs.coeff
    dc = add_id(dc,c_id_dict)
    dc.insert(0, 'variable', dc.index)
        
        ## Without scaling: get the conditional quantiles 
    qru = QuantileReg(depvar, indvars=regressors_avl,
                      quantile_list=ql, data=dall,
                      model=model,
                      scaling=False, alpha=0.1)

        ## Run the predictions on the full frame (estimates can differ)
    dcq = qru.cond_quant
    dcq = add_id(dcq, c_id_dict)
    
    ## Store the coefficients
    dci = qru.coeff; dci = add_id(dci, c_id_dict)
    dci.insert(0, 'variable', dci.index)
    dc.rename(columns={'coeff':'coeff_scale'},inplace=True)
    dc['coeff_noscale']=dci['coeff']
    dc=dc[['variable','horizon','quantile','coeff_scale','coeff_noscale','pval','lower','upper','R2_in_sample','normalized', 'Model']]

    exitcode=1
    return [dc,dcq,dci,exitcode]

def add_id(df, id_dict):
    """ Add identifiers variables to a pandas frame """
    variables_id = sorted(list(id_dict.keys()))
    for v, var in enumerate(variables_id):
        df.insert(v, var, id_dict[var])
    return(df)

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
            qfit = self.QModel(y, X).fit(q=tau, maxiter=2000, p_tol=1e-05)

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

