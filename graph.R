d1 <- read.csv('timings_2015-03-19.csv')
d1
library(ggplot2)
library(reshape2)
# we only want the important columns as the rest other two are fixed at 22,10000
d2 <- d1[,c('Iterations','Rtime','CUDAtime')]
d2
# convert from "wide" to "tall" format
d3 <- melt(d2, id='Iterations')
d3
qplot(Iterations, value, colour=variable, data=d3, log='xy', ylab='Time (Secs)', geom=c('line','point'))+ geom_line(size=3) + theme_grey(base_size = 28) +
  theme(axis.title.x = element_text(size = 28), axis.title.y = element_text(size = 28), axis.text = element_text(size = 20)) 
  
# dimensions in inches 
ggsave('performance_plot_2015-03-20.pdf', height=7, width=7)
# nb. can also save to say .png with dimensions also in inches, defaulting to 300dpi
#ggsave('performance_plot_2015-03-20.png', height=7, width=7)

d2$Ratio <- d2$Rtime/d2$CUDAtime
qplot(Iterations, Ratio, data=d2, log='x', geom=c('point','line'))
# ggsave('performance_ratio_plot_2015-03-22.pdf', height=7, width=7)
