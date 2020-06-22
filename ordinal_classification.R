library(tidyverse)

ordinal_coder = function(y) {
  
  num_classes = length(unique(y))
  
  new_y = as.data.frame(matrix(nrow = length(y), ncol = num_classes - 1))
  
  for (i in 1:length(y)) {
    arr = rep(0,num_classes-1)
    arr[ min(y[i],1):y[i] ] = 1
    new_y[i,] = rbind(arr)
  }
  
  return(new_y)
}

predict_ordinal = function(test_data, model_array) {
  
  num_classes = length(model_array) + 1
  
  class_probs = data.frame(p0 = rep(0, nrow(test_data)))
  
  class_probs[,'p0'] = 1 - predict(model_array[[1]], test_data, type="prob")$`1`
  
  for (class in 2:(num_classes - 1)) {
    
    class_probs[,paste0('p', class - 1)] = predict(model_array[[class-1]], test_data, type="prob")$`1` - predict(model_array[[class]], test_data, type="prob")$`1`
    
  }
  
  class_probs[,paste0("p", num_classes - 1)] = predict(model_array[[num_classes - 1]], test_data, type="prob")$`1`
  
  return(apply(class_probs, 1, which.max) - 1) # returns classifications as index of max probability
}
