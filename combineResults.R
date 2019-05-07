
library(dplyr)

directories <- commandArgs(trailingOnly = TRUE)

tgt <- c('results','configurations','models')

data <- lapply(tgt,function(dataType){
          lapply(directories, function(directory){
            read.csv(paste0(directory,'/',dataType,'.csv'), stringsAsFactors = FALSE)})
})

data <- lapply(data,bind_rows)

names <- data[[2]]$name
data[[2]] <- apply(data[[2]],2,function(col){
                      ifelse(is.na(col) | as.numeric(col) == 0,
                             0,
                             1)
   }) %>% as.data.frame()

data[[2]]$name <- names

data <- lapply(data, function(df){
                  nameCol <- names(df) == 'name'
                  names <- df[nameCol]
                  saveRDS(df[!nameCol], 'tee.rds')
                  values <- sapply(df[!nameCol],as.numeric)
                  cbind(names,values)
})

i <- 1
for(dataType in data){
   write.csv(data[[i]],paste0('results/', tgt[i], '.csv'),row.names = FALSE)
   i <- i + 1
}

