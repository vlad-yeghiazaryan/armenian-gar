/*

iMaPP_Fig5.do

Notes: This sample code demonstrates how to generate charts on the distributions
	   of the average LTV limit using the iMaPP database, as shown in Figure 5 
	   of Alam et al. (2019). 

	   Please locate this do file and iMaPP_Fig1234.xlsx in the folder 
	   where iMaPP_M.dta is located, which is produced by iMaPP_load.do. 
	   Please set the current directory to the folder when running this do file. 
	   	   
	   Input:  This code uses iMaPP_M.dta that is produced by iMaPP_load.do.
	   Output: This code saves Stata figures (emf files) and the table of summary
	           stats in a Word document (Average_LTV_stats_`today'.docx).
	   
	   Please see iMaPP_Fig123.do for Figures 1, 2, and 3 of Alam et al. (2019).
	   Please see iMaPP_Fig4.do for Figure 4 of Alam et al. (2019).		   
	   
Reference:
	   Alam, Zohair, Adrian Alter, Jesse Eiseman, Gaston Gelos, Heedon Kang, 
	   Machiko Narita, Erlend Nier, and Naixi Wang, 2019, "Digging Deeper 
	   – Evidence on the Effects of Macroprudential Policies from a New Database"
	   IMF Working Paper WP/19/66. (www.imf.org/iMaPP)

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
log using "iMaPP_Fig5_`today'.txt", text replace

// Set up
local flg_Fig_5    1 // 1 if you want to export LTV charts (iMaPP_Fig5a.emf and iMaPP_Fig5b.emf) for Fig 5
local flg_WordLTV  1 // 1 if you want to export the summary of LTV_average to a Word document. 

// Specify the end period
// e.g., this was set at "2016M12" for Fig 5 in Alam et al. (2019)
local End_date "2021M12" 

*===============================================================================
* Step 2: LTV distribution (Figure 5)
*===============================================================================
if `flg_Fig_5' == 1 {

	use "./iMaPP_M.dta", clear
	drop if ifscode == . 
	drop if AE      == .  // Curacao, whose AE/EMDE are not available
	sort  ifscode datem
	xtset ifscode datem

	// Keep only countires with LTV_average data
	egen sum_LTV_average = total(LTV_average), by(ifscode) missing
	gen flg_LTV_average  = (sum_LTV_average!=.)
	keep if flg_LTV_average == 1

    // Histogram of the average LTV limit, conditioning on the use of LTV limits (i.e., LTV_average < 100). 
	//   Please note that Netherelands is an exception, which has regulatory LTV limits 
	//   but at higher levels than 100 percent(its average LTV limit is at 102 percent as of Dec 2016).
	label variable LTV_average "Average LTV limit (As of `End_date')"
	hist LTV_average if LTV_average<100 & datem == tm(`End_date'), frequency width(5) kdensity kdenopts(lwidth(1)) 
		graph export "./iMaPP_Fig5a.emf", as(emf) replace
	gen AE1 = (AE==0)
	graph box LTV_average if LTV_average<100 & datem == tm(`End_date'), over(AE1, relabel(1 "AEs" 2 "EMDEs"))
		graph export "./iMaPP_Fig5b.emf", as(emf) replace

	// Stats of the average LTV limit 
	xtsum LTV_average if datem == tm(`End_date')
	xtsum LTV_average if datem == tm(`End_date') & LTV_average < 100
	
}

*===============================================================================
* Step 3: Table of Summary stats to a Word document (for your info)
*===============================================================================
if `flg_WordLTV' == 1 {

	// Install "xtsum2docx.ado", if it has not been installed yet 
	capture ssc install xtsum2docx

	// Name of the Word document
	local wordfile iMaPP_LTV_average_statistics.docx
	
	// Statistics at the end-2016, conditioning that it is lower than 100
	xtsum2docx LTV_average if LTV_average<100 & datem == tm(`End_date') using "./`wordfile'", replace ///
	 mean(%9.1f) min(%9.1f) max(%9.1f) p25(%9.1f) median(%9.1f) p75(%9.1f) obs(%9.0fc) xtn(%9.0fc) ///
	 order(mean min p25 median p75 max obs xtn) ///
	 title("Average LTV Limit (<100%) - All (as of `End_date')")
	xtsum2docx LTV_average if LTV_average<100 & datem == tm(`End_date') & AE == 1 using "./`wordfile'", append ///
	 mean(%9.1f) min(%9.1f) max(%9.1f) p25(%9.1f) median(%9.1f) p75(%9.1f) obs(%9.0fc) xtn(%9.0fc) ///
	 order(mean min p25 median p75 max obs xtn) ///
	 title("Average LTV Limit (<100%) - AEs (as of `End_date')")
	xtsum2docx LTV_average if LTV_average<100 & datem == tm(`End_date') & AE == 0 using "./`wordfile'", append ///
	 mean(%9.1f) min(%9.1f) max(%9.1f) p25(%9.1f) median(%9.1f) p75(%9.1f) obs(%9.0fc) xtn(%9.0fc) ///
	 order(mean min p25 median p75 max obs xtn) ///
	 title("Average LTV Limit (<100%) - EMDEs (as of `End_date')")
	 
}
***********  END *************
log close
exit
