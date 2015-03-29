# Load the shared library containing the log-likelihood
# function that runs through CUDA
library(Rcpp)
dyn.load("rcuda.so")

# Wrapper function to call the CUDA implementation of
# the log-likelihood function
cudaLog <- function(a, b, t, iterations) 
{
  .Call("cudaLog", a, b, t, iterations)
}

# This is the function in R that is implemented in the above CUDA
# function, it is used to verify that the results are the same
fn1 <- function(a, b, t) {
  N <- length(t)
  nv <- length(a)
  stopifnot(length(a)==length(b))
  TMP <- vector(mode='numeric', N)
  tmp <- vector(mode='numeric', nv)
  for (j in 1:N) {
    for (k in 1:nv)
      tmp[[k]] <- exp(a[[k]] + b[[k]]*t[[j]])
    TMP[[j]] <- log(sum(tmp))
  }
  sum(TMP)
}

ITER <- 100 	# The number of iterations to run
NVER <- 100 	# fake version vector length
DLEN <- 10    # length of fake data

cat('Number of Iterations: ', ITER, '\n')
cat('Vector size: ', NVER, '\n')

# Setting up a and b
a <- vector(mode='list', length=ITER)
b <- vector(mode='list', length=ITER)

set.seed(123) 		    # to make sure the fake data is the same each time!
data <- runif(DLEN) 	# fake data

# Used to hold the results from the R implementation
RES <- vector(mode='numeric', length=ITER)	 
# Used to hold the results from the CUDA implementation
RES1 <- vector(mode='numeric', length=ITER) 

# Filling a and b with fake data
for(i in 1:ITER)
{
  a[[i]] <- runif(NVER)
  b[[i]] <- runif(NVER)
}

# Running the R version of the function and timing it
T0 <- Sys.time()
for (i in 1:ITER) {
  RES[[i]] <- fn1(a[[i]], b[[i]], data)
}
T1 <- Sys.time()
TD <- T1-T0

cat('Time taken R: ', TD, units(TD), '\n')

# Running the CUDA version of the function and timing it
T2 <- Sys.time()
RES1 <- cudaLog(a, b, data, ITER)
T3 <- Sys.time()
TD1 <- T3-T2

cat('Time taken CUDA: ', TD1, units(TD1), '\n')

# Checking that the results are the same
cat("All Equal? ",all.equal(RES, RES1), '\n')