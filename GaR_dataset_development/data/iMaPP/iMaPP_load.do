/*

iMaPP_load.do

Notes: The iMaPP database was originally introduced by the following paper.
	 
	 Zohair Alam, Adrian Alter, Jesse Eiseman, Gaston Gelos, Heedon Kang, 
	 Machiko Narita, Erlend Nier, and Naixi Wang, 2019, "Digging Deeper 
	 – Evidence on the Effects of Macroprudential Policies from a New Database"
	 IMF Working Paper WP/19/66. (www.imf.org/iMaPP)
	
	When using this data for your research or article, we would appreciate it 
	if you could kindly acknowledge “the IMF’s integrated Macroprudential Policy 
	(iMaPP) Database, originally constructed by Alam et al. (2019)” 
	as the source information, citing the original paper as above. 
	
	This STATA code demonstrates (1) how to load the Monthly data from 
    the iMaPP database; (2) how to construct Quarterly data using a time 
	aggregation method (e.g., sum), which can be changed to your preferred method;
	and (3) save them in the STATA data format (i.e., in a dta file). Please set 
	the current directory to the folder where the iMaPP database is located.

	For more information, please see the TOC sheet of the iMaPP database and 
	the above-mentioned paper.

Last Updated: April 11, 2023

*/

*===============================================================================
* Step 1: Set up
*===============================================================================
clear all 
set maxvar 10000
set more off
set excelxlsxlargefile on
pause off
capture log close
				  
// Set the log file
local today : display %tdCYND date(c(current_date), "DMY")  // Date (YYYYMMDD)
log using "iMaPP_load_`today'.txt", text replace

// Version of the iMaPP database
local iMaPP "iMaPP_database-2023-4-11.xlsx"

// Set up
local flg_M = 1 // 1 if you want to compile a monthly data file (iMaPP_M.dta)
local flg_Q = 1 // 1 if you want to compile a quarterlyl data file (iMaPP_Q.dta)

*===============================================================================
* Step 2: Monthly data
*===============================================================================
if `flg_M' == 1 {
	*---------------------------------------------------------------------------
	* Step 2-1: Load the dummy-type indicators
	*---------------------------------------------------------------------------
	local sheet_list MaPP MaPP_T MaPP_L
	local flg_firsttime = 1
	foreach sheet of local sheet_list {
		*local sheet      MaPP_T
		*local flg_firsttime = 0
		di " Import the `sheet' sheet ======================================"
		import excel using "./`iMaPP'", sheet("`sheet'") cellrange (A1) firstrow clear

		// Time variable (datem)
		gen datem   = ym(Year, Month)
		format datem %tm
		label var datem "year-month (%tm)"

		// Time variable (dateq)
		gen dateq = qofd(dofm(datem))
		label var dateq "year-quarter (quarterly:%tq)"
		format dateq %tq

		local varlist ifscode AE EMDE CCB Conservation ///
			  Capital Capital_Gen Capital_HH Capital_Corp Capital_FX ///
			  LVR LLP LCG LCG_Gen LCG_HH LCG_Corp LoanR LoanR_HH LoanR_Corp LFC ///
			  LTV DSTI Tax Liquidity LTD LFX RR RR_FCD SIFI OT SUM_17	
		
		// De-string MaPP variables	
		local MaPP_17_list CCB Conservation Capital LVR LLP LCG LoanR LFC ///
						   LTV DSTI Tax Liquidity LTD LFX RR SIFI OT
		local MaPP_sub_list Capital_Gen Capital_HH Capital_Corp Capital_FX ///
					   LCG_Gen LCG_HH LCG_Corp LoanR_HH LoanR_Corp RR_FCD				
		local varlist_MaPP `MaPP_17_list' `MaPP_sub_list' SUM_17
		
		foreach var of local varlist_MaPP {
			if "`sheet'" == "MaPP_T" {
				local var `var'_T
			}
			if "`sheet'" == "MaPP_L" {
				local var `var'_L
			}		
			destring `var', ignore("NA") replace
		}
		// De-string other variables
		local varlist_other ifscode AE EMDE 
		foreach var of local varlist_other {
			destring `var', ignore("NA") replace
		}
		
		// Merge and save	
		if `flg_firsttime' == 1 {
			order Country iso3 iso2 ifscode AE EMDE Year Month datem dateq
			save "./iMaPP_M.dta", replace
			local flg_firsttime = 0
		} 
		else {
			di "Variable `sheet'"
			sort iso3 datem
			order Country iso3 iso2 ifscode AE EMDE Year Month datem dateq
			merge 1:1 iso3 datem using "./iMaPP_M.dta", nogenerate
			save "./iMaPP_M.dta", replace
		}		
	}

	// Data labels
	label var ifscode "IFS country codes"
	label var iso3    "ISO country codes (3 digits)"
	label var iso2    "ISO country codes (2 digits)"
	label var AE      "Advanced Economies (Yes:1, No:0), WEO 2018"
	label var EMDE    "Emerging Market and Developing Economies (Yes:1, No:0), WEO 2018"
	label var CCB     "Countercyclical buffers"
	label var Conservation "Capital conservation buffers"
	label var Capital      "Capital requirements"
	label var Capital_Gen  "Capital requirements: General"
	label var Capital_HH   "Capital requirements: Household sector targeted"
	label var Capital_Corp "Capital requirements: Corporate sector targeted"	
	label var Capital_FX   "Capital requirements: FX loans targeted"	
	label var LVR          "Leverage limits"
	label var LLP          "Loan loss provisions"
	label var LCG          "Limits on credit growth"
	label var LCG_Gen      "Limits on credit growth: General"
	label var LCG_HH       "Limits on credit growth: Household sector targeted"
	label var LCG_Corp     "Limits on credit growth: Corporate sector targeted"
	label var LoanR        "Loan restrictions"
	label var LoanR_HH     "Loan restrictions: Household sector targeted"
	label var LoanR_Corp   "Loan restrictions: Corporate sector targeted"
	label var LFC          "Restrictions on foreign currency loans"
	label var LTV          "Limits on the loan-to-value ratio"
	label var DSTI         "Limits on the debt-service-to-income or loan-to-income ratio"
	label var Tax          "Tax measures for macroprudential purposes"
	label var Liquidity    "Liquidity requirements"
	label var LTD          "Limits on the loan-to-deposit ratio"
	label var LFX          "Limits on the foreign exchange positions"
	label var RR           "Reserve requirements"
	label var RR_FCD       "Reserve requirements: foreign currency differentiated"
	label var SIFI         "Measures for the systemically important financial institutions"
	label var OT           "Other macroprudential measures"
	label var SUM_17       "Sum of the 17 policy-action dummy-type indicators (w/o subcategories)"

	// Save
	save "./iMaPP_M.dta", replace

	*---------------------------------------------------------------------------
	* Step 2-2: Load the average LTV limit (and the median LTV limit, FYI)
	*---------------------------------------------------------------------------
	local sheet      LTV_average
	local flg_firsttime = 0
	di " Import the `sheet' ======================================"
	import excel using "./`iMaPP'", sheet("`sheet'") cellrange (A1) firstrow clear
	label var LTV_average  "Average LTV limit"
	label var LTV_median   "(Supplemental info) Median LTV limit"
	
	// De-string LTV_average and LTV_median
    destring LTV_average, ignore("NA") replace
	destring LTV_median,  ignore("NA") replace
	
	// Time variable (datem)
	gen datem   = ym(Year, Month)
	format datem %tm
	label var datem "Time variable in Stata format (monthly:%tm)"

	// Time variable (dateq)
	gen dateq = qofd(dofm(datem))
	label var dateq "Time variable in Stata format (quarterly:%tq)"
	format dateq %tq

	// Merge and save	
	di "Variable LTV_average (and LTV_median, which is supplemental information)"
	sort iso3 datem
	order Country iso3 iso2 ifscode AE EMDE Year Month datem dateq
	merge 1:1 iso3 datem using "./iMaPP_M.dta", nogenerate
	save "./iMaPP_M.dta", replace
}

*===============================================================================
* Step 3: Quarterly data
*===============================================================================
if `flg_Q' == 1 {

	// Load the Monthly data
	if `flg_M' == 0 {
		use "./iMaPP_M.dta", clear
	}

	// List of dummy variables to be included in the Quarterly data
	local varlist CCB Conservation ///
		  Capital Capital_Gen Capital_HH Capital_Corp Capital_FX ///
		  LVR LLP LCG LCG_Gen LCG_HH LCG_Corp LoanR LoanR_HH LoanR_Corp LFC ///
		  LTV DSTI Tax Liquidity LTD LFX RR RR_FCD SIFI OT SUM_17

	// Time aggregation
	collapse (sum) `varlist' (mean) LTV_Qmean = LTV_average LTV_Qmean_med = LTV_median ///
	         (last) LTV_Qend = LTV_average LTV_Qend_med = LTV_median ///
			 (max)  LTV_Qmax = LTV_average LTV_Qmax_med = LTV_median, by(dateq Country ifscode iso2 iso3 AE EMDE)
		
	// Time variable (Year and Quarter)
	gen Year    = year(dofq(dateq))
	gen Quarter = quarter(dofq(dateq))
	label var Year "Year"
	label var Quarter "Quarter (1-4)"
	order Quarter, before(dateq)
	order Year, before(Quarter)
	
	// Data labels
	label var ifscode "IFS country codes"
	label var iso3    "ISO country codes (3 digits)"
	label var iso2    "ISO country codes (2 digits)"
	label var AE      "Advanced Economies (Yes:1, No:0), WEO 2018"
	label var EMDE    "Emerging Market and Developing Economies (Yes:1, No:0), WEO 2018"
	label var CCB     "Countercyclical buffers"
	label var Conservation "Capital conservation buffers"
	label var Capital      "Capital requirements"
	label var Capital_Gen  "Capital requirements: General"
	label var Capital_HH   "Capital requirements: Household sector targeted"
	label var Capital_Corp "Capital requirements: Corporate sector targeted"	
	label var Capital_FX   "Capital requirements: FX loans targeted"	
	label var LVR          "Leverage limits"
	label var LLP          "Loan loss provisions"
	label var LCG          "Limits on credit growth"
	label var LCG_Gen      "Limits on credit growth: General"
	label var LCG_HH       "Limits on credit growth: Household sector targeted"
	label var LCG_Corp     "Limits on credit growth: Corporate sector targeted"
	label var LoanR        "Loan restrictions"
	label var LoanR_HH     "Loan restrictions: Household sector targeted"
	label var LoanR_Corp   "Loan restrictions: Corporate sector targeted"
	label var LFC          "Restrictions on foreign currency loans"
	label var LTV          "Limits on the loan-to-value ratio"
	label var DSTI         "Limits on the debt-service-to-income or loan-to-income ratio"
	label var Tax          "Tax measures for macroprudential purposes"
	label var Liquidity    "Liquidity requirements"
	label var LTD          "Limits on the loan-to-deposit ratio"
	label var LFX          "Limits on the foreign exchange positions"
	label var RR           "Reserve requirements"
	label var RR_FCD       "Reserve requirements: foreign currency differentiated"
	label var SIFI         "Measures for the systemically important financial institutions"
	label var OT           "Other macroprudential measures"
	label var SUM_17       "Sum of the 17 policy-action dummy-type indicators (w/o subcategories)"
    label var LTV_Qmean    "Average LTV limit: quarter average of monthly LTV_average"
    label var LTV_Qend     "Average LTV limit: end-quarter value of monthly LTV_average"
    label var LTV_Qmax     "Average LTV limit: quarter max of monthly LTV_average"
    label var LTV_Qmean_med "(Supplemental info) Median LTV limit: quarter average of monthly LTV_median"
    label var LTV_Qend_med  "(Supplemental info) Median LTV limit: end-quarter value of monthly LTV_median"
    label var LTV_Qmax_med  "(Supplemental info) Median LTV limit: quarter max of monthly LTV_median"	
	
	// Save
	sort ifscode dateq
	save "./iMaPP_Q.dta", replace
}
***********  END *************
log close
exit
