# Calculating the growth
def calc_growth_rate(series, horizon=4, yearfreq=4, method_growth='cpd'):
  if method_growth=='cpd':
    growth = cum_gr(series, horizon=horizon, yearfreq=yearfreq)
  elif method_growth=='yoy':
    growth = yoy_gr(series, horizon, yearfreq=yearfreq)
  elif method_growth=='level':
    growth = series.shift(-horizon)
  else:
    growth = None
  return growth

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
