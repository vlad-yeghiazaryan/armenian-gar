cd "~/Desktop/GaR-project/FCyI" //update the path
clear *
local country Armenia  // Armenia

qui import excel "`country'_data_initial.xlsx", sheet("data") firstrow clear
qui gen Country="`country'"
global Y Consumerstockdiff Mortgagestockdiff  Corporatestockdiff Propertypricesyoygrowtrate TotalloansGDPYoY CredittoGDPGAP TradeaccountdeficitGDP Creditspreadhouseholds Creditspreadnonfinancialcor
qui drop Country
local n=_N

foreach var in $Y {
sort `var'
gen i=_n
tsset i
kdensity `var', kernel(gaussian) generate(cdf at) nograph n(`n')
cumul cdf, gen(K_`var')
drop i cdf at
}
qui keep Date K_*
tsset Date

foreach var in $Y {
qui rename K_`var' `var'
// twoway (tsline `var'), ttitle(.) ttitle(, size(zero)) ytitle(, size(vsmall)) tlabel(#5, labsize(vsmall) angle(forty_five)) title("`var'", size(vsmall)) name(`var', replace)
}

// graph combine $Y, name(graph1, replace) title("Kernel Transformation for `country' Data")
// graph drop $Y
// qui export excel using "`country'_data.xlsx", sheet("data") sheetmodify firstrow(variables)

// qui import excel "`country'_data_initial 1.xlsx", sheet("Y") firstrow clear
// qui export excel using "`country'_data.xlsx", sheet("Y") sheetmodify firstrow(variables)
