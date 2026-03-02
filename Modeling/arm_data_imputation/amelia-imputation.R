# setup
library(dplyr)
library(readr)
library(Amelia)
library(ggplot2)

# function for taking the mean of values across dfs and creating 1 df
combineImputations <- function(imp, arm_data){
  imputations <- lapply(imp, function(df) {
    rownames(df) <- df$Date
    df <- df[, !colnames(df) %in% c("date", "country")]  # drop some specific columns by name
    return(df)
  })
  mean_imputed_data = Reduce(`+`, imputations) / length(imputations)
  mean_imputed_data$date = arm_data$date
  mean_imputed_data$country = arm_data$country
  return(mean_imputed_data)
}

# loading data
# arm_data_consistent.csv
arm_data <- read_csv("../../data/arm_data/arm_data_consistent.csv", 
                          col_types = cols(date = col_date(format = "%Y-%m-%d")))
arm_data = as.data.frame(arm_data)
arm_vars = colnames(arm_data)[c(-1)]

# polytime = 2
# lags='tariff'
# logs='tariff'
# intercs = TRUE
# empri = .01 * nrow(panel_dataset)
# lags, leads, logs, sqrts, lgstc, 
# noms, ords, 

# using amelia to fill in the missing data with m=5 dataframes
arm_data.out.poly <- amelia(arm_data, m = 5, ts = "date",
                            polytime = 1,
                            p2s = 2)

# saving the resulting imputations
save(arm_data.out.poly, file = "arm_data_imputations_P1.RData")

# Loading the imputed panel
load("arm_data_imputations_P1.RData")

# Overdispersed Starting Values: run the EM algorithm from multiple,
# dispersed starting values and check their convergence.
disperse(arm_data.out.poly, dims = 1, m = 5)

# checking the density of the imputed vs actual series
# plot(price.out.simple, which.vars = price_columns)
plot(arm_data.out.poly, which.vars = arm_vars)

# combining (into a single dataframe)
mean_imputed_arm_data <- combineImputations(arm_data.out.poly$imputations, 
                                            arm_data)

# plot some finalized series
ggplot(mean_imputed_arm_data, aes(x=date, y="Real GDP_PctC_1y")) + geom_line()

# exporting the imputations
write.csv(mean_imputed_arm_data, file = "../../data/arm_data/arm_data_filled.csv", row.names = FALSE)
