# Classifier.R — only these two lines:
model_list <- readRDS("model_objs.rds")
list2env(model_list, envir = .GlobalEnv)