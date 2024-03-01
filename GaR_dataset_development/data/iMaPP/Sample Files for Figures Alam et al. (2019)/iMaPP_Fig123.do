/*

iMaPP_Fig123.do

Notes: This sample code demonstrates how to generate charts on the use of 
	   macroprudential policy, as shown in Figure 1, 2, and 3 of 
	   Alam et al. (2019). 
	   
	   Please locate this do file and iMaPP_Fig1234.xlsx in the folder 
	   where iMaPP_M.dta is located, which is produced by iMaPP_load.do. 
	   Please set the current directory to the folder when running this do file. 
	   
	   Input:  This code uses iMaPP_M.dta that is produced by iMaPP_load.do.
	   Output: This code saves results in selected sheets of iMaPP_Fig1234.xlsx.
	           Please see the charts in sheet "Fig1," "Fig2," and "Fig3" 
			   on the same Excel file.
	
	   Please see iMaPP_Fig4.do for Figure 4 of Alam et al. (2019).
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
log using "iMaPP_Fig123_`today'.txt", text replace

// Set up
local flg_iMaPP_plus 1 // 1 if you want to compile (iMaPP_plus.dta) with the cumulative sum measures.
local flg_Fig_123    1 // 1 if you want to export (iMaPP_Fig1234.xlsx) for Fig 1-3

// Specify the end period (optional)
// e.g., this was set at "keep if datem <= tm(2016M12)"
//       which was the sample period used in Alam et al. 2019
// Default setting would be empty.
local keep_if_datem_before_SpecTime 

// Load iMaPP data (both dummy-type and average LTV data)
use "./iMaPP_M.dta", clear
drop if ifscode == . 
drop if AE      == .  // Curacao, whose AE/EMDE are not available
xtset ifscode datem

*===============================================================================
* Step 2: Lists of MaPP indicators
*===============================================================================
levelsof iso3, local(Nat_Code)

local MaPP_17_list CCB Conservation LVR LLP Capital LTV DSTI ///
	               LCG LoanR LFC Tax Liquidity LTD LFX RR SIFI OT
local MaPP_16_list CCB Conservation LVR LLP Capital LTV DSTI ///
	               LCG LoanR LFC Tax Liquidity LTD LFX  SIFI OT
	  
local MaPP_17_list_T 
local MaPP_17_list_L
foreach x of local MaPP_17_list {
	local MaPP_17_list_T  `MaPP_17_list_T' `x'_T
	local MaPP_17_list_L  `MaPP_17_list_L' `x'_L
}

local MaPP_sub_list Capital_Gen Capital_HH Capital_Corp Capital_FX ///
	               RR_FCD RR_NRD LCG_Gen LCG_HH LCG_Corp  LoanR_HH LoanR_Corp

*===============================================================================
* Step 3: Cumulative sum of MaPP indicators
*===============================================================================
if `flg_iMaPP_plus' == 1 {
	local flg_firsttime = 1
	foreach x of local MaPP_17_list {
		preserve 
		
		di " Cusum of `x' ======================================"
		// Reshape to wide.
		keep iso3 datem `x'  `x'_T  `x'_L
		reshape wide    `x'  `x'_T  `x'_L, i(datem) j(iso3) string
		
		// Running sum:  the sum of the first through jth observations on x
		local Cusum_list_A   
		local Cusum_list_T  
		local Cusum_list_L 
		local AE_list
		foreach k of local Nat_Code {
			gen `x'_cs_A`k' = sum(`x'`k')
			gen `x'_cs_T`k' = sum(`x'_T`k')
			gen `x'_cs_L`k' = sum(`x'_L`k')
			local Cusum_list_A `Cusum_list_A' `x'_cs_A`k' 
			local Cusum_list_T `Cusum_list_T' `x'_cs_T`k' 
			local Cusum_list_L `Cusum_list_L' `x'_cs_L`k'
		} 
		
		// Reshape back to long.
		keep datem `Cusum_list_A' `Cusum_list_T' `Cusum_list_L' 	 
		reshape long `x'_cs_A `x'_cs_T `x'_cs_L, i(datem) j(iso3) string
		rename `x'_cs_A  `x'_cs
		gen    `x'_cs_act   = `x'_cs_T + `x'_cs_L
		label var `x'_cs       "Cumulative sum: `x'"
		label var `x'_cs_T     "Cumulative sum: `x'_T"
		label var `x'_cs_L     "Cumulative sum: `x'_L"	
		label var `x'_cs_act   "Cumulative # of actions: `x'_cs_T + `x'_cs_L"	
		
		// Merge and save	
		if `flg_firsttime' == 1 {
			sort iso3 datem
			merge 1:1 iso3 datem using "./iMaPP_M.dta", nogenerate
			save "./iMaPP_Plus.dta", replace
			local flg_firsttime = 0
		} 
		else {
			sort iso3 datem
			merge 1:1 iso3 datem using "./iMaPP_Plus.dta", nogenerate
			save "./iMaPP_Plus.dta", replace
		}		
		
		restore	
	}
}

*===============================================================================
* Step 4: Preverence and Frequency by instruments (Figure 1-3)
*===============================================================================
if `flg_Fig_123' == 1 {
	// Load dataset produced in  Step 2
	use "./iMaPP_Plus.dta", clear
	drop if ifscode == . 
	drop if AE      == .  // Curacao, whose AE/EMDE are not available	
	sort  ifscode datem
	xtset ifscode datem
	
	// for debug
	local MaPP_17_list CCB Conservation LVR LLP Capital LTV DSTI ///
					   LCG LoanR LFC Tax Liquidity LTD LFX RR SIFI OT
	local MaPP_16_list CCB Conservation LVR LLP Capital LTV DSTI ///
					   LCG LoanR LFC Tax Liquidity LTD LFX  SIFI OT	
	
	// Specify the end period (optional)
	di "`keep_if_datem_before_SpecTime'"
	`keep_if_datem_before_SpecTime'
	
	// Set frequency
	local frequency "A"  // A: Annual, Q: Quarterly, M: Monthly
				
	*------------------------------------------------------
	* 3.1 Use dummy (Fig 1-2)
	*------------------------------------------------------
	gen Sum_17_use = 0
	local use_list
	foreach x of local MaPP_17_list {
		gen       `x'_use     = (`x'_cs_act > 0)
		replace   Sum_17_use  = Sum_17_use + `x'_use
		label var `x'_use     "1 if ever used (`x'_cs_act > 0)"
		label var Sum_17_use  "sum of _use for all 17 instruments"	
		local use_list   `use_list' `x'_use
	}
	gen Sum_16_use = 0
	foreach x of local MaPP_16_list {
		replace  Sum_16_use = Sum_16_use + `x'_use
		label var Sum_16_use  "sum of _use for all (except for RR)"	
	}
	// Flag for the use of any instruments
	gen MaPP17_use = (Sum_17_use > 0)
	gen MaPP16_use = (Sum_16_use > 0)
	label var MaPP17_use  "1 if ever used any of the 17 MaPP instruments"	
	label var MaPP16_use  "1 if ever used any MaPP (except for RR)"	
	
	// Stats for All
	preserve 
		local sheet = "ALL"
		gen N   = AE + EMDE

		// First, collapse by datem to get the group sum at a monthly frequency
		collapse (sum) `use_list'  N MaPP16_use MaPP17_use (mean) Year dateq, by(datem)
		if "`frequency'" == "A" {
			// Then, collapse by Year to get annual series
			collapse (mean) N (lastnm) MaPP16_use MaPP17_use `use_list', by(Year)
			order N MaPP16_use MaPP17_use `use_list' 
			local time = "Year"
		}
		if "`frequency'" == "Q" {
			// Then, collapse by Year to get quarterly series
			collapse (mean) N (lastnm) MaPP16_use MaPP17_use `use_list', by(dateq)
			order N MaPP16_use MaPP17_use `use_list' 
			local time = "dateq"
		}		
		
		// Export to an Excel file
		export excel `time' N  *_use ///
			   using "./iMaPP_Fig1234.xlsx", sheet("`sheet'_`frequency'") firstrow(variables) missing("NA") sheetreplace
	restore

	//  Stats for AE and EMDE
	preserve 
		gen N   = AE + EMDE

		// First, collapse by datem and AE to get the group sum at a monthly frequency
		collapse (sum) `use_list'  N MaPP16_use MaPP17_use (mean) Year dateq, by(datem AE)
		if "`frequency'" == "A" {
			// Then, collapse by Year to get annual series
			collapse (mean) N (lastnm) MaPP16_use MaPP17_use `use_list' , by(Year AE)
			order N MaPP16_use MaPP17_use `use_list' 
			local time = "Year"
		}
		if "`frequency'" == "Q" {
			// Then, collapse by Year to get quarterly series
			collapse (mean) N (lastnm) MaPP16_use MaPP17_use `use_list' , by(dateq AE)
			order N MaPP16_use MaPP17_use `use_list' 
			local time = "dateq"
		}		
			
		// Reshape to wide.
		reshape wide N MaPP16_use MaPP17_use `use_list' , i("`time'") j(AE)

		// Export to an Excel file
		forvalues group = 0/1 {
			if `group' == 0 {
				local sheet = "EMDE"
			}
			if `group' == 1 {
				local sheet = "AE"
			}
			export excel `time' N`group'  *_use`group'  ///
				   using "./iMaPP_Fig1234.xlsx", sheet("`sheet'_`frequency'") firstrow(variables) missing("NA") sheetreplace
		}	
	restore
		
	*----------------------------------------------------------
	* 3.2 Frequency (# of actions per period: 2010M1-; Fig3)
	*----------------------------------------------------------
	// Number of actions for each instrument
	local list `MaPP_17_list'
	local act_list
	foreach x of local list {	
		gen       `x'_act   = `x'_T + `x'_L
		label var `x'_act   "# of actions: `x'_T + `x'_L"	
		local act_list   `act_list' `x'_act
	}
	// Number of actions for all 17 instruments
	gen     SUM_17_act = SUM_17_T + SUM_17_L
	replace SUM_17_act = (SUM_17_act > 0)
	label var SUM_17_act  "1 if any use of 17 instruments"		
	
	// List of the number-of-action variables
	local act_list   `act_list' SUM_17_act	
	
	// Stats for All
	preserve
		local sheet = "ALL"
		
		// Focus on the recent years when many instruments have been introduced.
		keep if datem >= tm(2010M1) 
		
		// Specify the end period (optional)
		di "`keep_if_datem_before_SpecTime'"
		`keep_if_datem_before_SpecTime'
				
		// First, collapse by country to get the country-specific average #s of actions per month
		collapse (mean) `act_list', by(iso3)
		gen N = 1
		// Second, collapse to get the average # of actions per month
		collapse (sum) N (mean) `act_list'
		foreach x of local act_list {
			// Average number of actions per year
			replace `x' = `x'*12
		}
		
		// Export to an Excel file
		export excel N `act_list' ///
			   using "./iMaPP_Fig1234.xlsx", sheet("Act_`sheet'") firstrow(variables) missing("NA") sheetreplace
	restore

	//  Stats for AE and EMDE
	preserve 
		local sheet = "AEEM"
		
		// Focus on the recent years when many instruments have been introduced.
		keep if datem >= tm(2010M1)	
		
		// Specify the end period (optional)
		di "`keep_if_datem_before_SpecTime'"
		`keep_if_datem_before_SpecTime'
		
		// First, collapse by country to get the country-specific average #s of actions per month
		collapse (lastnm) AE (mean) `act_list', by(iso3)
		gen N = 1
		// Second, collapse to get the average # of actions per month, by country group
		collapse (sum) N (mean) `act_list', by(AE)
		foreach x of local act_list {
			// Average number of actions per year
			replace `x' = `x'*12
		}
			
		// Export to an Excel file
		export excel AE N `act_list' ///
			   using "./iMaPP_Fig1234.xlsx", sheet("Act_`sheet'") firstrow(variables) missing("NA") sheetreplace
	restore	
			
}

***********  END *************
log close
exit
