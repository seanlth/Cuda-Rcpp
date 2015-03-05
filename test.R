library(Rcpp)
dyn.load("rcuda.so")

cudaLog <- function(a, b, t, iterations) 
{
  .Call("cudaLog", a, b, t, iterations)
}

fn1 <- function(a, b, t) {
  N <- length(t)
  nv <- length(a)
  stopifnot(length(a)==length(b))
  TMP <- vector(mode='numeric', N)
  for (j in 1:N) {
    for (k in 1:nv)
      tmp.k <- exp(a[[k]] + b[[k]]*t[[j]])
    TMP[[j]] <- log(sum(tmp.k))
  }
  sum(TMP)
}

ITER <- 10000
NVER <- 100 # fake version vector length

cat('Number of Iterations: ', ITER, '\n')
cat('Vector size: ', NVER, '\n')

a <- vector(mode='list', length=ITER)
b <- vector(mode='list', length=ITER)

set.seed(123) # to make sure the fake data is the same each time!
data <- runif(10) # fake data
RES <- vector(mode='numeric', length=ITER)
RES1 <- vector(mode='numeric', length=ITER) 

for(i in 1:ITER)
{
  a[[i]] <- runif(NVER)
  b[[i]] <- runif(NVER)
}

T0 <- Sys.time()
for (i in 1:ITER) {
  RES[[i]] <- fn1(a[[i]], b[[i]], data)
}
T1 <- Sys.time()
TD <- T1-T0

cat('Time taken R: ', TD, units(TD), '\n')

T2 <- Sys.time()
RES1 <- cudaLog(a, b, data, ITER)
T3 <- Sys.time()
TD1 <- T3-T2

cat('Time taken CUDA: ', TD1, units(TD1), '\n')

cat("All Equal? ",all.equal(RES, RES1), '\n')