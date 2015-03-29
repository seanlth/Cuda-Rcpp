ITER <- 1 #c(1,2,5, 10,20,50, 100,200,500,1000, 2000, 5000)
NVER <- 22 #c(1,2,5, 10,20,50, 100,200,500)
DLEN <- c(1,2,5, 10,20,50, 100,200,500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000)

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
  tmp <- vector(mode='numeric', nv)
  for (j in 1:N) {
    for (k in 1:nv)
      tmp[[k]] <- exp(a[[k]] + b[[k]]*t[[j]])
    TMP[[j]] <- log(sum(tmp))
  }
  sum(TMP)
}

sink("outfile2.txt")

for(i in ITER)
{
  for(n in NVER)
  {
    for(d in DLEN)
    {
      cat("Num interations : ", i, "\n")
      cat("Len vectors : ", n, "\n")
      cat("Len data : ", d, "\n")
      
      a <- vector(mode='list', length=i)
      b <- vector(mode='list', length=i)
      set.seed(123)
      data <- runif(d)
      
      RES <- vector(mode='numeric', length=i)	 
      RES1 <- vector(mode='numeric', length=i) 
      
      for(j in 1:i)
      {
        a[[j]] <- runif(n)
        b[[j]] <- runif(n)
      }
      
      T0 <- Sys.time()
      for (j in 1:i) {
        RES[[j]] <- fn1(a[[j]], b[[j]], data)
      }
      T1 <- Sys.time()
      TD <- T1-T0
      
      cat('Time taken R: ', TD, units(TD), '\n')
      
      T2 <- Sys.time()
      RES1 <- cudaLog(a, b, data, i)
      T3 <- Sys.time()
      TD1 <- T3-T2
      
      cat('Time taken CUDA: ', TD1, units(TD1), '\n')
      cat("All Equal? ",all.equal(RES, RES1), '\n')
    }
  }
}
