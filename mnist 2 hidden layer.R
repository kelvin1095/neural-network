library("tidyverse")
set.seed(25)

## Control parameters
layer_input <- 784
layer_1 <- 24
layer_2 <- 16
layer_output <- 10
alpha <- 0.1


## Activation functions
relu <- function(x){
  pmax(x, matrix(0, ncol(x), nrow(x)))
}

drelu <- function(x){
  x > 0
}

sigmoid <- function(x){
  1/(1+exp(-x))
}

softmax <- function(x){
  exp(x)/sum(exp(x))
}


## Import Data
train <- read_csv('mnist_train.csv')
test <- read_csv('mnist_test.csv')
n <- nrow(train)
m <- ncol(train)

## Formatting
## Take the label
train_number <- data.matrix(train[,1])
test_number <- data.matrix(test[,1])

## Rescale pixels to be between 0 and 1
train_input <- data.matrix(train[,2:m])/255
test_input <- data.matrix(test[,2:m])/255

## The output should be a vector of 0 with a 1 corrosponding to the number
t_function <- function(x){
  target <- rep(0, 10)
  target[x+1] <- 1
  target
}
train_target <- train_number %>% apply(1, t_function) %>% t()
test_target <- test_number %>% apply(1, t_function) %>% t()

output_label <- c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

## Split data
t_input <- input[1:n_train,]
t_target <- target[1:n_train,]
v_input <- input[(n_train + 1):n,]
v_target <- target[(n_train + 1):n,]

output_label <- c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


## Create random w1, w2 (weight matrix), b1, b2 (intercepts)
initialise_weights <- function(layer_input, layer_1, layer_2, layer_output){
  w1 <- matrix(runif(layer_input*layer_1, min = -0.5, max = 0.5), layer_input, layer_1)
  w2 <- matrix(runif(layer_1*layer_2, min = -0.5, max = 0.5), layer_1, layer_2)
  w3 <- matrix(runif(layer_2*layer_output, min = -0.5, max = 0.5), layer_2, layer_output)
  b1 <- matrix(runif(layer_1, min = -0.5, max = 0.5), 1, layer_1)
  b2 <- matrix(runif(layer_2, min = -0.5, max = 0.5), 1, layer_2)
  b3 <- matrix(runif(layer_output, min = -0.5, max = 0.5), 1, layer_output)
  
  return(list(w1 = w1,
              b1 = b1,
              w2 = w2,
              b2 = b2,
              w3 = w3,
              b3 = b3))
}

## Forward propagation
forward_propagation <- function(input, param){
  Z1 <- input %*% param$w1 + rep(param$b1, each = nrow(input))
  A1 <- relu(Z1)
  
  Z2 <- A1 %*% param$w2 + rep(param$b2, each = nrow(input))
  A2 <- relu(Z2)
  
  Z3 <- A2 %*% param$w3 + rep(param$b3, each = nrow(input))
  A3 <- apply(Z3, 1, softmax) %>% t()
  
  return(list(Z1 = Z1,
              A1 = A1,
              Z2 = Z2,
              A2 = A2,
              Z3 = Z3,
              A3 = A3))
}

## Backward propagation
backward_propagation <- function(input, param, output, target, alpha){
  n_train <- nrow(input)
  
  dZ3 <- (target - output$A3)
  dW3 <- t(output$A2) %*% dZ3
  db3 <- colSums(dZ3)
  
  dZ2 <- dZ3 %*% t(param$w3) * drelu(output$Z2)
  dW2 <- t(output$A1) %*% dZ2
  db2 <- colSums(dZ2)
  
  dZ1 <- dZ2 %*% t(param$w2) * drelu(output$Z1)
  dW1 <- t(input) %*% dZ1
  db1 <- colSums(dZ1)
  
  w1 <- param$w1 + alpha * dW1/n_train
  b1 <- param$b1 + alpha * db1/n_train
  w2 <- param$w2 + alpha * dW2/n_train
  b2 <- param$b2 + alpha * db2/n_train
  w3 <- param$w3 + alpha * dW3/n_train
  b3 <- param$b3 + alpha * db3/n_train
  
  return(list(w1 = w1,
              b1 = b1,
              w2 = w2,
              b2 = b2,
              w3 = w3,
              b3 = b3))
}

## MSE
MSE_fn <- function(target, output){
  sum((target - output$A3)^2)/nrow(target)
}

accuracy_fn <- function(target, output){
  sum(max.col(target) == max.col(output$A3))/nrow(target)
}

## Initialise weights
param <- initialise_weights(layer_input, layer_1, layer_2, layer_output)

## Train the model
for(i in 1:500){
  output <- forward_propagation(train_input, param)
  
  param <- backward_propagation(train_input, param, output, train_target, alpha)
  
  ## Iteration Statistics
  if(0 == (i %% 10)){
    output <- forward_propagation(test_input, param)
    MSE <- MSE_fn(test_target, output)
    accuracy <- accuracy_fn(test_target, output)
    message('Iteration ', str_pad(i, 4, pad = "0"), 
            ' Prediction accuracy ', round(accuracy, 4) %>% as.character() %>% str_pad(6, pad = "0", side = 'right'), 
            ' MSE ', MSE)
  }
}


output <- forward_propagation(test_input, param)
prediction <- (max.col(output$A3) - 1) %>% sapply(t_function) %>% t()

#prediction
#test_target

# Define a function that operates on two corresponding elements
my_function <- function(x, y) {
    a <- which(x == 1)
    b <- which(y == 1)
    true_positive <- union(a, b) %>% length()
    false_positive <- setdiff(a, b) %>% length()
    false_negative <- setdiff(b, a) %>% length()

    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1_score = 2*(precision * recall)/(precision + recall)
    
    return(list(true_positive = true_positive,
               false_positive = false_positive,
               false_negative = false_negative,
               precision = precision,
               recall = recall,
               f1_score = f1_score))  # Multiply corresponding elements from two columns
}

# Apply the function to corresponding elements of two columns from different matrices
my_function(prediction[,1], test_target[,1])
my_function(prediction[,2], test_target[,2])
my_function(prediction[,3], test_target[,3])
my_function(prediction[,4], test_target[,4])
my_function(prediction[,5], test_target[,5])
my_function(prediction[,6], test_target[,6])
my_function(prediction[,7], test_target[,7])
my_function(prediction[,8], test_target[,8])
my_function(prediction[,9], test_target[,9])
my_function(prediction[,10], test_target[,10])
