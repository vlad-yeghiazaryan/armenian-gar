# setup
library(dplyr)
library(readr)
library(Amelia)
library(ggplot2)

# function for taking the mean of values across dfs and creating 1 df
combineImputations <- function(imp, panel_dataset){
  imputations <- lapply(imp, function(df) {
    rownames(df) <- df$Date
    df <- df[, !colnames(df) %in% c("date", "country")]  # Drop some specific columns by name
    return(df)
  })
  mean_imputed_panel_dataset = Reduce(`+`, imputations) / length(imputations)
  mean_imputed_panel_dataset$date = panel_dataset$date
  mean_imputed_panel_dataset$country = panel_dataset$country
  return(mean_imputed_panel_dataset)
}

# loading data
# panel_dataset_consistent.csv
panel_dataset <- read_csv("../data/working_data/panel_dataset_consistent.csv", 
                          col_types = cols(date = col_date(format = "%Y-%m-%d")))
panel_dataset = as.data.frame(panel_dataset)
panel_vars = colnames(panel_dataset)[c(-1, -2)]

# panel_dataset_normalized.csv
panel_dataset.normalized <- read_csv("../data/working_data/panel_dataset_normalized.csv", 
                                     col_types = cols(date = col_date(format = "%Y-%m-%d")))
panel_dataset.normalized = as.data.frame(panel_dataset.normalized)
panel_vars.normalized = colnames(panel_dataset.normalized)[c(-1, -2)]

# polytime = 2
# lags='tariff'
# logs='tariff'
# intercs = TRUE
# empri = .01 * nrow(panel_dataset)
# lags, leads, logs, sqrts, lgstc, 
# noms, ords, 

# using amelia to fill in the missing data with m=5 dataframes
# panel_dataset.out.poly
# panel_dataset.norm.poly
panel_dataset.out.poly <- amelia(panel_dataset, m = 5, cs="country", ts = "date",
                                 empri = .01 * nrow(panel_dataset),
                                 polytime = 1,
                                 intercs = TRUE,
                                 p2s = 2)

# saving the resulting imputations
save(panel_dataset.out.poly, file = "panel_dataset_imputations_PIC.RData")

# Loading the imputed panel
load("panel_dataset_imputations_PIC.RData")
load("panel_dataset_imputations_PICN.RData")


# plotting the individual time series
# par(mfrow = c(1, 1))
plot_var <- "Equity_index"
tscsPlot(panel_dataset.out.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)
tscsPlot(panel_dataset.norm.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)

# plotting the individual time series
# par(mfrow = c(1, 1))
plot_var <- "GDP_real_PctC4"
tscsPlot(panel_dataset.out.poly, cs="Ukraine", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)
plot_var <- "GDP_real"
tscsPlot(panel_dataset.norm.poly, cs="Ukraine", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)


# plotting the individual time series
# par(mfrow = c(1, 1))
plot_var <- "Bank_liabilities_MC_PctC4"
tscsPlot(panel_dataset.out.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)
plot_var <- "Bank_liabilities_MC"
tscsPlot(panel_dataset.norm.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)

# plotting the individual time series
# par(mfrow = c(1, 1))
plot_var <- "Credit_GDP_ratio_C4"
tscsPlot(panel_dataset.out.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)
plot_var <- "Credit_GDP_ratio"
tscsPlot(panel_dataset.norm.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)

# Overimputing: Involves sequentially treating each of the observed values as 
# if they had actually been missing.
overimpute(panel_dataset.out.poly, var = 'GDP_real_PctC4')

# Overdispersed Starting Values: run the EM algorithm from multiple,
# dispersed starting values and check their convergence.
disperse(panel_dataset.out.poly, dims = 1, m = 5)

# checking the density of the imputed vs actual series
# plot(price.out.simple, which.vars = price_columns)
plot(panel_dataset.out.poly, which.vars = panel_vars)
plot(panel_dataset.norm.poly, which.vars = panel_vars.normalized)

# combining (into a single dataframe)
mean_imputed_panel_dataset <- combineImputations(panel_dataset.out.poly$imputations, 
                                                panel_dataset)
mean_imputed_panel_dataset.normalized <- combineImputations(panel_dataset.norm.poly$imputations, 
                                                            panel_dataset.normalized)

# plot some finalized series
ukr = mean_imputed_panel_dataset.normalized[mean_imputed_panel_dataset.normalized$country == 'Ukraine',]
ggplot(ukr, aes(x=date, y=GDP_real)) + geom_line()

# exporting the imputations
write.csv(mean_imputed_panel_dataset, file = "../data/working_data/panel_dataset_filled.csv", row.names = FALSE)
write.csv(mean_imputed_panel_dataset.normalized, file = "../data/working_data/panel_dataset_nf.csv", row.names = FALSE)
