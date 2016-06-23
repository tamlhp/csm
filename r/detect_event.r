#fn <- "test.sample.en.series"
fn <- "test.en.series"
#fn <- "raw_data"
#fn = "test.en.wwat.series"
#fn = "test.en.topic"
my_data <- read.table(paste(fn,".csv",sep=""), header=FALSE, sep="\t", colClasses=c("POSIXct",NA))
#write.table(my_data, paste(fn,".out.csv",sep=""), sep="\t", row.names = FALSE, col.names=FALSE)
library(AnomalyDetection)
#data(raw_data)
res = AnomalyDetectionTs(my_data, max_anoms=0.02, direction='both', plot=TRUE)
res$plot

library(BreakoutDetection)
colnames(my_data) <- c("timestamp", "count")
res2 = breakout(my_data, min.size=24, method='multi', beta=.001, degree=1, plot=TRUE)
res2$plot