# setup
library(dplyr)
library(readr)
library(Amelia)

# loading data
panel_dataset <- read_csv("../data/working_data/panel_dataset_consistent.csv", 
                          col_types = cols(date = col_date(format = "%Y-%m-%d")))
panel_dataset = as.data.frame(panel_dataset)
panel_vars = colnames(panel_dataset)[c(-1, -2)]

# polytime = 2
# lags='tariff'
# logs='tariff'
# intercs = TRUE
# empri = .01 * nrow(panel_dataset)
# lags, leads, logs, sqrts, lgstc, 
# noms, ords, 

# using amelia to fill in the missing data with m=5 dataframes
panel_dataset.out.poly <- amelia(panel_dataset, m = 5, cs="country", ts = "date",
                                 empri = .01 * nrow(panel_dataset),
                                 polytime = 1,
                                 intercs = TRUE,
                                 p2s = 2)

# saving the resulting imputations
save(panel_dataset.out.poly, file = "panel_dataset_imputations_PIC.RData")

# Loading the imputed panel
load("panel_dataset_imputations_PIC.RData")


# plotting the individual time series
# par(mfrow = c(1, 1))
plot_var <- "Equity_index"
tscsPlot(panel_dataset.out.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)

# plotting the individual time series
# par(mfrow = c(1, 1))
plot_var <- "GDP_real_hz_4_growth"
tscsPlot(panel_dataset.out.poly, cs="Saudi Arabia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)

# plotting the individual time series
# par(mfrow = c(1, 1))
plot_var <- "Bank_liabilities_MC_PctC_4"
tscsPlot(panel_dataset.out.poly, cs="Armenia", 
         main = paste(plot_var, " (PI)"),
         var = plot_var)

# Overimputing: Involves sequentially treating each of the observed values as 
# if they had actually been missing.
overimpute(panel_dataset.out.poly, var = 'GDP_real_growth')

# Overdispersed Starting Values: run the EM algorithm from multiple,
# dispersed starting values and check their convergence.
disperse(panel_dataset.out.poly, dims = 1, m = 5)

# checking the density of the imputed vs actual series
# plot(price.out.simple, which.vars = price_columns)
plot(panel_dataset.out.poly, which.vars = panel_vars)

# combining (into a single dataframe) and exporting the imputations
imputations <- lapply(panel_dataset.out.poly$imputations, function(df) {
  rownames(df) <- df$Date
  df <- df[, !colnames(df) %in% c("date", "country")]  # Drop some specific columns by name
  return(df)
})

mean_imputed_panel_dataset = Reduce(`+`, imputations) / length(imputations)
mean_imputed_panel_dataset$date = panel_dataset$date
mean_imputed_panel_dataset$country = panel_dataset$country
write.csv(mean_imputed_panel_dataset, file = "../data/working_data/panel_dataset.csv", row.names = FALSE)
write.csv(mean_imputed_panel_dataset, file = "../data/results/panel_dataset.csv", row.names = FALSE)
