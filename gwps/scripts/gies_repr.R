library(reticulate)
library(pcalg)
np <- import('numpy')

library("optparse")
 
option_list <- list(
  make_option(c("--data"), type="character", default=NULL, 
              help="dataset file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="out.txt", 
              help="output file name [default= %default]", metavar="character")
); 
 
opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

filename <- opt$data
df <- np$load(filename)
p <- dim(df[['obs']])[2]
X <- df[['obs']]

for (i in 0:(p - 1)) {
    X <- rbind(X, df[[toString(i)]])
}

target_vec <- list(integer(0))
for (i in 0:(p-1)) {
    target_vec <- append(target_vec, as.integer(i + 1))
}

n_obs <- dim(df[['obs']])[1]
intervention_indicator <- rep(1, n_obs)
for (i in 0:(p - 1)) {
    n_i <- dim(df[[toString(i)]])[1]
    intervention_indicator <- c(intervention_indicator, rep(i + 2, n_i))

}
score <- new("GaussL0penIntScore", X, target_vec, intervention_indicator)
gies.fit <- gies(score)

W_est <- as(gies.fit$repr, 'matrix')
W_est[W_est] <- 1

np$save(opt$out, W_est)
