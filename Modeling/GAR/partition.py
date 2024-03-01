   
## 3rd-party modules
import numpy as np
import pandas as pd
from datetime import datetime as date 

## Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression   ## PLS

def retropolated_PCA(data, groups_dict, target, horizon=4, method_growth='cpd', method='PCA', benchcutoff=0.2):
    '''
    Main run function for step 1, partition. Function to generate retropolated partition in a time period.
    '''
    # setup for input data partitioning
    first_valid = data.apply(lambda x: x.first_valid_index())
    last_valid = data.apply(lambda x: x.last_valid_index())
    start_index = max([min([first_valid[value] for value in values]) for key, values in groups_dict.items()])
    end_index = min([min([last_valid[value] for value in values]) for key, values in groups_dict.items()])
    dall = data.set_index(data['date'], drop=False)
    sdate = data['date'][start_index].to_pydatetime()
    edate = data['date'][end_index].to_pydatetime()

    # ------------------------
    # Run the partition
    # ------------------------
    # Some data treatment for missing data at the end
    # dall = dall.fillna(method='ffill').copy()
    dall = dall[(dall['date']>=sdate) & (dall['date']<=edate)]
    
    if (horizon < 4) and (method_growth=='yoy'):
        dall = dall.iloc[4-horizon-1:]
        sdate = dall.index.values[0]

    # Using dependent variable as benchmark, although it is an extra copy, the code for benchmark is written for accepting any bench mark variable, and keeping this will give the flexibility for futrue using other variables.
    bench='benchvar'
    dall[bench] = dall[target]

    # Generating all cutoffs in the period, sorted from latest to earliest
    [cutoffs,complete_group] = gen_cutoff(dall=dall, groups_dict=groups_dict,startdate=sdate, enddate=edate)

    if (cutoffs==-1):
        raise Exception("In the given time period some groups are completely empty. No feasible partition can be made.")
    if len(cutoffs)==0:
        raise Exception("No data in the cutoff period")

    # Generating the parition for the latest cutoff            
    [dpr, retroload] = p_cutoff(dall, groups_dict,cutoffs[0], bench, method, benchcutoff, None, False)
    for i in range(1,len(cutoffs)):
        [dpn, dln] = p_cutoff(dall,groups_dict,cutoffs[i],bench,method, benchcutoff, None, False)    
        dpr = retropolation(dfearly=dpn, dflate=dpr, complete_early=complete_group[i],groups_dict=groups_dict)
    retroload['cutoff'] = sdate
    if method=='PLS':
        retroload=retroload[['variable','cutoff','loadings','group','vip']]
    else:
        retroload=retroload[['variable','cutoff','loadings','group','variance_ratio']]
        
    # Compute the zscore for the final frame to makes them consistent
    group_vars = [x for x in groups_dict.keys()]
    for group in group_vars:
        dpr[group] = zscore(dpr[group])

    dpr.index.name=None
    dall.index.name=None
    retroframe = dpr.merge(dall[['date', target]], on=['date'], how='left')
    retroframe.index=retroframe['date']
    retroframe.index.name=None
    retroframe.reset_index(drop=True, inplace=True)
    
    return retroframe, retroload

# Zscore correction
def zscore(series):
    return((series - series.mean())/series.std(ddof=0))

###############################################################################
# Given two frame of signle country retroplate late frame to early frame, 
# return the retroplated frame
def retropolation(dfearly, dflate, complete_early, groups_dict):
###############################################################################
    # Select the data of interest
    # This part can be removed as it shoud be done outside of the function.
    group_vars = list(groups_dict.keys())
    all_vars = ['date'] + group_vars

    de = dfearly.loc[:,all_vars].copy()
    dl = dflate.loc[:,all_vars].copy()

    # Sort the frames
    de = dfearly.sort_values(by=['date'], ascending=True)
    dl = dflate.sort_values(by=['date'], ascending=True)

    # For every country, compute the reverse growth rate based on early data
    ###############################################################################
    # Compute the reverse delta (from future to now, data inverted)
    for pvar in group_vars:
        rgr_n = '{}_rgr'.format(pvar)

        # Need to normalize: compute the zscore, per country     
        de[pvar] = zscore(de[pvar]) 
        dl[pvar] = zscore(dl[pvar])
    
        # Compute the delta, per country (pay attention to the order, future second)
        de[rgr_n] = (lambda x: x - x.shift(-1))(de[pvar])
        dl[rgr_n] = lambda x: x - x.shift(-1)(dl[pvar])   

    ###############################################################################
    # Index creation using the reverse delta
    # Dulani's trick: sum for small numbers, growth rate for large number !!
    ###############################################################################

    # 1. Identify the missing dates from the late frame
    # dec = de.loc[de.country==pays,:]
    # dmc = dm.loc[dm.country==pays,:]
    # dlc = dl.loc[dl.country==pays,:]

    # From late to middle
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
        dng = dng.sort_values(by='date', ascending=0)
        gr_cum = '{}_cum_rgr'.format(group)
        dng_f = dng.merge(ef[['date', gr_cum]], on=['date'], how='left')
        #If group in the early frame is complete, no retroplation is needed. Use the value in early group.
        if group in complete_early:
            dng_f[group]=ef[group].values
        else:
            dng_f[group] = dng_f[gr_cum] + start_val
        dng_f.index=dng_f['date'].values
        dng_f.index.name=None
        mgr_frames_list.append(dng_f)
        
    # Merge the new groups into a early augmented frame
    dea = mgr_frames_list[0]
    for frame in mgr_frames_list[1:]:
        dea = pd.merge(dea, frame, on=['date'])
    dea.index=dea['date'].values
    dea.index.name=None

    ## Merge late, early augmented
    d_complete = pd.concat([dl[all_vars], dea[all_vars]],axis='index')
    d_complete=d_complete.sort_values(by=['date'])
    dfearly=dfearly.sort_values(by=['date'])
    
    ## complete group fix
    for group in complete_early:
        d_complete[group]=dfearly[group]

    return d_complete

## Function to generate partition cutoff points and completed groups
## at the coressponding cutoff ponit. Completed group will not be retropolated.
def gen_cutoff (dall="default",groups_dict={}, startdate=date(year=1,month=1,day=1), enddate=date(year=9999,month=12,day=31),):
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
            #########################################################
            # There exists a complete emptry group, the cutoff date is not applicable
                print(d, key)
                for v in values:
                    print(v,dall[v].first_valid_index())
                print("In the given time period some groups are complete empty. No feasible partition can be made")
                return -1,-1
            else:
                if complete_key:
                    tmp_c_key.append(key)
        complete_groups.append(tmp_c_key)
            
            
    return dates,complete_groups

###############################################################################
# Fuction to do partion for one cutoff time. Return partion and loading
###############################################################################
def add_id(df, id_dict):
    """ Add identifiers variables to a pandas frame """
    variables_id = sorted(list(id_dict.keys()))
    for v, var in enumerate(variables_id):
        df.insert(v, var, id_dict[var])
    return(df)

def p_cutoff(dall,groups_dict,cutoff,bench,method, benchcutoff, PLStarget, saveim=False):

    df = dall.loc[dall.date >= cutoff].copy()

    partition_dict = groups_dict
    variables=[]
    label_dict={}
    for key, values in partition_dict.items():
        variables.extend(values)
        for e in values:
            label_dict[e]=e
    

    #country=df['country'].values[0]
    c_id_dict = {'cutoff' : cutoff.strftime("%Y-%m-%d"), 
                 'variables': repr(variables)}
                #'variables': repr(gv.indvars_dict[country])}
    try:
        df.loc[:,'var_benchmark'] = df[bench]

        dfbch = df.dropna(subset=['var_benchmark'], axis=0, how='any')[['var_benchmark']].copy()
        ## Define the benchmark for the partition
        threshold = dfbch['var_benchmark'].quantile(benchcutoff) # Lower growth regime
        df.loc[:,'benchmark'] = (dfbch['var_benchmark'] < threshold)
        df['benchmark']=df['benchmark'].fillna(method='ffill')

        if method=='LDA':
            p = Partition(df, partition_dict,
                          reduction='LDA', benchmark='benchmark', PLStarget=PLStarget)
        elif method=='PCA':
            p = Partition(df, partition_dict,
                          reduction='PCA', benchmark=None , PLStarget=PLStarget)
        elif method=='PLS':
            p = Partition(df, partition_dict,
                          reduction='PLS', benchmark=None , PLStarget=PLStarget)
            

        dp = p.partition # Run the partition on the full frame

        
        dp = add_id(dp, c_id_dict)
        dp.loc[:,'date'] = dp.index
        dp.index.name=None
        ## Loading from the partitioning
        dl = p.loading; dl = add_id(dl, c_id_dict)
        dl.insert(0, 'variable_o', dl.index)
        dl.loc[:,'variable'] = dl.variable_o.apply(lambda x : label_dict[x])
        return [dp,dl]
        
#    # Customize the exception behaviour    
    except:
        #exc.args += (cutoff)
        print('partition failed!')
        return -1

###############################################################################
# Data partitioning
###############################################################################
class Partition(object):
    """ 
    Partition dataset using either supervised or unsupervised data reduction

    Inputs:
    - data: the dataset to reduce
    - groups_dict: groups of variables to perform the reduction along
    - reduction: either PCA or LDA. If LDA needs to provide a benchmark (str)
    - benchmark: Name of the variable to supervise the LDA reduction with (str)

    Outputs:
    - loadings: the loadings of the reduction (1 if only one variable)
    - partition: results of the partitioning

    Usage:
    Partition(df, gv.groups_dict, reduction='LDA', benchmark='benchmark')

    """
    __description = "Data partitioning using dimensionality reduction"
    __author = "Romain Lafarguette, IMF/MCM, rlafarguette@imf.org"

    ## Initializer
    def __init__(self, data, groups_dict, reduction='PCA',
                 benchmark=None,PLStarget=None):

        ## Parameters
        self.reduction = reduction
        self.benchmark = benchmark
        self.PLStarget = PLStarget
        ## Clean the dataset according to the type of reduction
        if self.reduction == 'LDA':
            if isinstance(self.benchmark, str):
                dc = data.dropna(subset=[self.benchmark], axis=0, how='any')
                #dc = data.dropna(axis=0, how='any').copy()
                self.data = dc.dropna(axis=1, how='any').fillna(method='ffill').copy()
            else:
                raise ValueError('Need a benchmark with supervised reduction')
        elif self.reduction == 'PCA':
            self.data = data.dropna(axis=0, how='all').dropna(axis=1,how='any').fillna(method='ffill').copy()
            
        elif self.reduction == "PLS":
            self.data = data.dropna(axis=0, how='all').dropna(axis=1,how='any').fillna(method='ffill').copy()
            
        else:
            raise ValueError('Reduction parameter misspecified')

        ## Remove constant columns (create problem in the partitioning)
        self.data = self.data.loc[:, self.data.apply(pd.Series.nunique) != 1]
        
        ## Populate the groups only with the variables available in the frame
        self.var_dict = {k:[x for x in groups_dict[k] if x in self.data.columns]
                         for k in groups_dict.keys()}

        ## Estimate the fit
        if self.reduction == 'PCA':
            self.partition_fit_group, self.loading = self.__partition_fit_PCA()

            ## For consistency with the LDA object, rename some instances
            for group in sorted(list(self.partition_fit_group.keys())):
                setattr(self.partition_fit_group[group], 'fit',
                        self.partition_fit_group[group].fit_transform)
            
        elif self.reduction == 'LDA':
            self.partition_fit_group, self.loading = self.__partition_fit_LDA()
            
        elif self.reduction == "PLS":
           self.partition_fit_group, self.loading = self.__partition_fit_PLS()

        else:
            raise ValueError('Reduction parameter misspecified')
            


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

    def __partition_fit_LDA(self):
        """ Run the data partitioning using Linear Discriminant Analysis """
        groups = sorted(list(self.var_dict.keys()))
        lda_fit_group = dict()
        loadings_frame = list()
        
        for group in groups:
            var_list = self.var_dict[group]
            if len(var_list) > 1: # Run the partition
                # Partitionning
                dg = self.data.loc[:, var_list].copy()
                
                X = scale(dg) # Need to scale the variables before partitioning
                y = self.data.loc[:, self.benchmark].values
                
                ## Fit the LDA using the benchmark
                lda_fit = LDA(n_components=1).fit(X, y)
                lda_fit_group[group] = lda_fit

                ## Store the loadings
                dl = pd.DataFrame(lda_fit.coef_, index=['loadings'],
                                  columns=var_list).transpose()
        
                dl['variance_ratio']=lda_fit.explained_variance_ratio_[0]
                dl['group'] = group
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
        return((lda_fit_group, dloading))        
    
    def __partition_fit_PLS(self):
        """ Run the data partitioning using Principal Component Analysis """
        groups = sorted(list(self.var_dict.keys()))
        pls_fit_group = dict()
        loadings_frame = list()
        
        for group in groups:
            var_list = self.var_dict[group]
            if len(var_list) > 1: # Run the partition
                # Partitionning
                plsdepvar=self.PLStarget[group]
                plsavlreg=[e for e in var_list if e not in self.PLStarget[group]]
                
                ## Fit the PLS
                pls = PLS_DA(plsdepvar, plsavlreg,self.data)         
                pls_fit = pls.fit
                pls_fit_group[group] = pls_fit

                ## Store the loadings
                dl = pls.summary
                dl['group'] = group
                dl['variable'] = dl.index
                loadings_frame.append(dl)
                
            elif len(var_list) == 1: # Loadings are 1
                dl = pd.DataFrame(index=var_list)
                dl['loadings'] = 1
                dl['vip']=1
                dl['group'] = group
                dl['variable'] = var_list[0]
                loadings_frame.append(dl)

            else: # Empty group: no loading
                dl = pd.DataFrame(columns=['loadings', 'group', 'variable'])
                dl['loadings'] = np.nan
                dl['vip']=np.nan
                dl['group'] = group
                dl['variable'] = np.nan
                loadings_frame.append(dl)

        dloading = pd.concat(loadings_frame)

        # Return the fit method and the associated loadings                
        return((pls_fit_group, dloading))   
    
    def partition_data(self, dataframe):
        """ Return the aggregated data """
        # From the previous step, extract the fitting for each group
        groups = sorted(list(self.var_dict.keys()))
        pfit = self.partition_fit_group # Either PCA or LDA

        ## Prepare to store the data and the loadings
        da = pd.DataFrame(index=dataframe.index)
        
        for group in groups:
            var_list = self.var_dict[group]
            if len(var_list) > 1: # Use the loadings from the partition fit
                dg = dataframe.loc[:, var_list].copy()
                
                # Scale the variables
                X = scale(dg) 
                              
                ## Generate the data using the partitioning fit
                if self.reduction=='PLS':
                    Y = scale(dataframe.loc[:, self.PLStarget[group]].copy())     
                    da[group] = pfit[group].fit_transform(X,Y)[0]                    
                else:
                    da[group] = pfit[group].transform(X)
                                    
                
            elif len(var_list) == 1: # Simply keep the variable as it is
                da[group] = dataframe.loc[:, var_list[0]]
        
            else: # Empty group
                da[group] = np.nan

        return(da)

def pls_reduction(depvars, regvars, df):
    assert isinstance(depvars, list), 'Dependent variable(s) should be in list'
    avl_regs = [x for x in regvars if x in df.columns]
    pls_series = PLS_DA(depvars, avl_regs, df).component
    return(pls_series)

def num_days(dates_tuple):
    """ Return the number of days in a tuple """
    min_ = pd.to_datetime(min(dates_tuple))
    max_ = pd.to_datetime(max(dates_tuple))
    return((max_ - min_).days)

###############################################################################
#%% PLS Discriminant Analysis Class Wrapper
###############################################################################
class PLS_DA(object):
    """ 
    Data reduction through PLS-discriminant analysis and variables selection 

    Parameters
    ----------
    dep_vars : list; list of dependent variables
    reg_vars : list; list of regressors variables
    data : pandas df; data to train the model on
    num_vars : 'all', integer; number of variables to keep, ranked by VIP
        if 'all': keep all the variables
    
    Return
    ------
    first_component : the first component of the PLS of the Xs reduction
    output_frame : frame containing the variables and their transformation
    summary_frame : frame with the results of the model (loadings, vip, R2)

    """
    __description = "Partial Least Squares with variables selection"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"

    #### Class Initializer
    def __init__(self, dep_vars, reg_vars, data, num_vars='all'):

        #### Attributes
        self.dep_vars = dep_vars
        self.reg_vars = reg_vars
        self.df = data.dropna(subset=self.reg_vars)

        ## Put parametrized regression as attribute for consistency
        self.pls1 = PLSRegression(n_components=1, scale=True) # Always scale

        ## Unconstrained fit: consider all the variables 
        self.ufit = self.pls1.fit(self.df[self.reg_vars],
                                  self.df[self.dep_vars])

        ## Return the component and summary of the unconstrained model
        ## To save computation time, run it by default for both models        
        self.component_unconstrained = self.__component(self.ufit,
                                                        self.dep_vars,
                                                        self.reg_vars, self.df)

        self.target_unconstrained = self.__target(self.ufit,
                                                  self.dep_vars,
                                                  self.reg_vars, self.df)

        self.summary_unconstrained = self.__summary(self.ufit, self.dep_vars,
                                                    self.reg_vars, self.df)

        ## Variables selection
        if num_vars == 'all': # Unconstrained model: constrained is identical
            self.top_vars = self.reg_vars # The best variables are the full set
            self.fit = self.ufit
            self.component = self.component_unconstrained
            self.target = self.target_unconstrained
            self.summary = self.summary_unconstrained
            
        elif num_vars > 0: ## Constrained model
            self.num_vars = int(num_vars)
            
            ## Identify the most informative variables from the unconstrained
            self.top_vars = list(self.summary_unconstrained.sort_values(
                by=['vip'], ascending=False).index[:self.num_vars])

            ## Now run the constrained fit on these variables
            self.cfit = self.pls1.fit(self.df[self.top_vars],
                                      self.df[self.dep_vars])

            ## Return the main attributes, consistent names with unconstrained
            self.fit = self.cfit
            
            self.component = self.__component(self.cfit, self.dep_vars,
                                              self.top_vars, self.df)
            
            self.target = self.__target(self.cfit, self.dep_vars,
                                        self.top_vars, self.df)

            self.summary = self.__summary(self.cfit, self.dep_vars,
                                          self.top_vars, self.df)
                      
        else:
            raise ValueError('Number of variables parameter misspecified')

        
    #### Internal class methods (start with "__")
    def __vip(self, model):
        """ 
        Return the variable influence in the projection scores
        Input has to be a sklearn fitted model
        Not available by default on sklearn, so it has to be coded by hand
        """
        ## Get the score, weights and loadings
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape

        ## Initialize the VIP
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = [(w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)]
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        return(vips)

    def __summary(self, fit, dep_vars, reg_vars, df):
        """
        Return the summary information about the fit
        """
        
        ## Store the information into a pandas dataframe
        dr = pd.DataFrame(reg_vars, columns=['variable'], index=reg_vars)
        dr['loadings'] = fit.x_loadings_ # Loadings
        dr['vip'] = self.__vip(fit) ## Variable importance in projection
        dr['score'] = fit.score(df[reg_vars],df[dep_vars]) # Score
        
        ## Return the sorted summary frame
        return(dr.sort_values(by=['vip'], ascending=False))
    
    ## Write short ancillary functions to export the results into pandas series
    def __component(self, fit, dep_vars, reg_vars, df):
        """
        Return the first component of the fit
        """
        comp = fit.fit_transform(df[reg_vars], df[dep_vars])[0]
        comp_series = pd.Series(comp.flatten(), index=self.df.index)
        return(comp_series)

    def __target(self, fit, dep_vars, reg_vars, df):
        """
        Return the target of the fit (reduced in case of multiple variables)
        """
        target = fit.fit_transform(df[reg_vars], df[dep_vars])[1]
        target_series = pd.Series(target.flatten(), index=self.df.index)
        return(target_series)

    
    #### Standard class methods (no "__")
    def predict(self, dpred):
        """ 
        Apply the dimension reduction learned on new predictors
        Input:
            - dpred: Pandas frame with the predictors 

        Output:
            - Reduced dataframe using the same loadings as estimated in-sample
 
        """
        
        ## Need to select exactly the predictors which have been estimated
        dp = dpred[self.top_vars].dropna()

        ## Run the projection
        dproj = pd.Series(self.fit.predict(dp).flatten(), index=dp.index)

        ## Scaling pb: prediction and fit don't match in sample (they should) !
        # Create the in-sample prediction
        dproj_in = pd.Series(self.fit.predict(self.df[self.top_vars]).flatten(),
                             index=self.df.index)
        
        # Adjust based on the in-sample projection !
        mean_adj =  dproj_in.mean() - self.component.mean()
        scale_adj = dproj_in.std()/self.component.std()

        # Nicely adjusted ! 
        dproj_mod = (dproj-mean_adj)/scale_adj
        
        return(dproj_mod)
