library(tidyverse)
library(quantreg)
library(glue)

qr_R1 <- function(model_1, model_2) {
  rho <- function(u, tau=model_1$tau) u * (tau - (u < 0))
  V_hat <- sum(rho(model_1$resid, model_1$tau))
  V0 <- sum(rho(model_2$resid, model_2$tau))
  R1 <- 1 - V_hat/V0
  R1
}

file <- "gar_dataset.csv"
data <- read.csv(file)

explanatory <- names(data)[c(3, 5, 6, 12, 13:15)]
outcome <- names(data)[length(names(data))]

rhs <- paste(explanatory, collapse = "+")
formula <- paste(outcome, rhs, sep = "~")
formula <- as.formula(formula)

y <- as.matrix(data[, outcome])
X <- as.matrix(cbind(const = 1, data[, explanatory]))
X_no_const <- X[, 2:ncol(X)]

# Main model
tau =  0.1
rq_model_const <- rq('real_y_ms_hz_4 ~ 1', tau, data)
rq_model_main <- rq(formula, tau=tau, data=data)
R1_main <- qr_R1(rq_model_main, rq_model_const)
R1_main

nConstrains <- 2
R <- matrix(0, 
            nrow = nConstrains, 
            ncol = ncol(X))
r <- matrix(0, nrow=nConstrains, ncol=1)

R[1, 2] = 1 
R[2, 2] = -1 
r[1, 1] = -1
r[2, 1] = -1

# The constrained model
rq_model_constrained <- rq(formula, tau=tau, R=R, r=r, method='fnc', data=data)
R1_constrained <- qr_R1(rq_model_constrained, rq_model_const)
R1_constrained

# Overview
summary_table <- cbind(main = rq_model_main$coefficients, 
                      constrained = rq_model_constrained$coefficients)

plot(y, type = "b", col="blue")
# lines(rq_model_main$fitted.values, type = "b", col="green")
lines(rq_model_constrained$fitted.values, type = "b", col="red")

print(summary_table)
print(glue("R1 main: {round(R1_main, 5)}"))
print(glue("R1 constrained: {round(R1_constrained, 5)}"))



  


