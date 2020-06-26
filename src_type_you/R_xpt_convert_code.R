setwd("~/Desktop/data type_you/raw-XPT")

library(foreign)
#filenames = dir(pattern="*.XPT")
#for (i in filenames){
#  mydata <- read.xport(i)
#  write.csv(mydata,  [[i]])
#}


#files <- list.files(path="~/Desktop/data type_you/raw-XPT", pattern="*.XPT")
#datalist <- lapply(files, function(x) {
#  t <- read.xport((x)) # load file
#})

#for( i in 1:length(files)){
#  write.csv(datalist[[i]] , paste0("~/Desktop/data type_you/interim-tocsv", files[[i]]), 
#          row.names=F)
#}



mydata <- read.xport(("TCHOL_J_NHANES_Total_Cholesterol_2017.XPT"))
write.csv(mydata, file = "~/Desktop/data type_you/interim-tocsv/TCHOL_J_NHANES_Total_Cholesterol_2017.csv", row.names = FALSE)

