
library(rjson)
library(magrittr)
library(tibble)
library(stringr)
library(dplyr)

# ================================
# Make a model table and results table out of each entry

getTables <- function(model){

   results <- tibble(fallout = model$test_fallout,
                     recall = model$test_recall,
                     precision = model$test_precision,
                     f1 = model$test_f1,
                     name = model$name)

   model <- list(C = model$C,
             n_features = model$features,
             time_total = model$time_total,
             time_preproc = model$time_preproc,
             time_vect = model$time_vect,
             time_tuning = model$time_tuning,
             time_crossval = model$time_crossval,
             name = model$name)

   list(results = results,
        model = model)
   }

# ================================

# Read and name model entries

resultFiles <- list.files('results')
resultFiles <- resultFiles[str_detect(resultFiles,'.json')]

models <- lapply(resultFiles ,function(jsonfile){
   readLines(paste0('results/',jsonfile)) %>% fromJSON()
   }) 

i <- 1
modelNames <- str_remove(resultFiles, '\\.json')
modelsRes <- list()
for(m in models){
   m$name <- modelNames[i] 
   i <- i + 1
   modelsRes[[m$name]] <- m
}
models <- modelsRes

# Make a dataset of config term occurrence

configSchema <- lapply(models,function(m) m$config) %>% unlist() %>% unique()

terms <- lapply(modelsRes, function(model){

             appears <- lapply(configSchema, function(configTerm){
                       as.numeric(configTerm %in% model$config)})

             names(appears) <- configSchema
             appears})


configTerms <- bind_rows(terms)
configTerms$name <- modelNames

# Get model tables

modelTables <- models %>% 
   lapply(getTables)

# Get results and model characteristics out

results <- lapply(modelTables, function(tbl){
          tbl$results}) %>% bind_rows()
models <- lapply(modelTables, function(tbl){
          tbl$model}) %>% bind_rows()

# ================================
# Writeout

write.csv(results,'results/results.csv', row.names = FALSE)
write.csv(models,'results/models.csv', row.names = FALSE)
write.csv(configTerms,'results/configurations.csv', row.names = FALSE)
