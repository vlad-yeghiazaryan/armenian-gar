/*

iMaPP_Fig4.do

Notes: This sample code demonstrates how to generate a chart on the evolution
       of macroprudential policy using the iMaPP database, as shown in Figure 4 
	   of Alam et al. (2019). Please note that the chart created by this code
	   looks slightly different from Figure 4 of Alam et al. (2019), 
	   because this code uses the full sample while Figure 4 uses 
	   the restricted sample whose quarterly data on household debt is available. 
	   Since household debt data are proprietary, we do not include it 
	   in this sample code.

	   Please locate this do file and iMaPP_Fig1234.xlsx in the folder 
	   where iMaPP_Q.dta is located, which is produced by iMaPP_load.do. 
	   Please set the current directory to the folder when running this do file. 
	   	   
	   Input:  This code uses iMaPP_Q.dta that is produced by iMaPP_load.do.
	   Output: This code saves results in sheet "Fig4_data" of iMaPP_Fig1234.xlsx.
	           Please see the chart in sheet "Fig4" on the same Excel file.
			   
	   Please see iMaPP_Fig123.do for Figures 1, 2, and 3 of Alam et al. (2019).
	   Please see iMaPP_Fig5.do for Figure 5 of Alam et al. (2019).
	   
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
log using "iMaPP_Fig4_`today'.txt", text replace

*===============================================================================
* Step 2: MaPP over time (Figure 4)
* Note: For Figure 4 of Alam et al. (2019), we also plot average credit growth. 
*===============================================================================
// Load iMaPP data (Quarterly)
use "./iMaPP_Q.dta", clear
drop if ifscode == . 
drop if AE      == .  // Curacao, whose AE/EMDE are not available
sort  ifscode dateq
xtset ifscode dateq	

// Construct the sum of all 17 instrument indicators (allsum)
local MaPP_17_list CCB Conservation LVR LLP Capital LTV DSTI ///
				   LCG LoanR LFC Tax Liquidity LTD LFX RR SIFI OT
// Generate the cumulative sum over the current and the past three quarters
gen allsum_yoy_s = 0
foreach x of local MaPP_17_list {
	gen `x'_yoy_s = `x' + L.`x' + L2.`x' + L3.`x' 
	replace allsum_yoy_s = allsum_yoy_s + `x'_yoy_s
}

// Take the sum of all actions by country group
collapse (sum) allsum_ = allsum_yoy_s, by(dateq AE EMDE)
*collapse (sum) allsum_ = allsum_yoy_s (mean) d4lnrhhdebt_w1_ = d4lnrhhdebt_w1, by(dateq AE EMDE)
sort AE dateq
tsline allsum_ if AE == 1
tsline allsum_ if AE == 0

// Reshape the dataset to long
gen 	C_group = "AE" if AE == 1
replace C_group = "EM" if AE == 0
drop EMDE AE
reshape wide allsum_ , i(dateq) j(C_group) string
*reshape wide allsum_ d4lnrhhdebt_w1_, i(dateq) j(C_group) string
label var allsum_AE "# of net tightening (AE, sum)"
label var allsum_EM "# of net tightening (EMDE, sum)"	
*label var d4lnrhhdebt_w1_AE "Credit growth yoy (AE, mean)"
*label var d4lnrhhdebt_w1_EM "Credit growth yoy (EMDE, mean)"	

// Export to an Excel file
export excel dateq allsum_AE allsum_EM ///
	   using "./iMaPP_Fig1234.xlsx", sheet("Fig4_data") firstrow(variables) missing("NA") sheetreplace	
		   
***********  END *************
log close
exit

