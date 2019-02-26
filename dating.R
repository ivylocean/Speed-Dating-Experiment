setwd("D:/RData/")
require(dplyr)
speed = tbl_df(read.csv("Speed Dating Data.csv"))
colnames <- colnames(speed)
n = dim(speed)[1]
p = dim(speed)[2]
index <- apply(apply(speed, 2, is.na), 2, sum)/n > 0.1
dating <- speed[ , !index]

#deleted some uninterested variables
dating <- subset(dating, select = -c(iid, id, idg, wave , partner, pid, field, 
                                   undergra, mn_sat, tuition, from, career,
                                   dec_o, dec))

## LASSO
require(Matrix)
require(glmnet)
require(caret)
set.seed(1)
inTrain <- createDataPartition(y = dating$match, p = 0.7, list = FALSE)
training <- st_dating[inTrain,]
testing <- st_dating[-inTrain,]

dating_mm <- model.matrix(match ~ 0+., data = training)
rown <- rownames(dating_mm)
rown <- as.integer(rown)
match_l <- as.vector(dating$match[rown])
datingglm <- glmnet(dating_mm, match_l, family = "binomial")
plot(datingglm, xvar = "lambda")
cv.datingglm <- cv.glmnet(dating_mm, match_l, family = "binomial")
plot(cv.datingglm)
best_lambda <- cv.datingglm$lambda.min
datingglm <- glmnet(dating_mm, match_l, family="binomial", lambda = best_lambda)
coef(datingglm)

# After LASSO Variable Selection
dating_dr <- subset(speed, 
                    select = c(gender, condtn, order, pf_o_att, attr_o, race, 
                               dining, concerts, met, match))
dating_dr <- na.omit(dating_dr)
st_dating_dr <- scale(dating_dr)
attach(st_dating_dr)
# split data
require(caret)
require(class)
set.seed(1)
inTrain <- createDataPartition(y = dating_dr$match, p = 0.7, list = FALSE)
training <- dating_dr[inTrain,]
testing <- dating_dr[-inTrain,]
train_match <- match[inTrain]
test_match <- match[-inTrain]

# split data
set.seed(1)
inTrain <- createDataPartition(y = st_dating_dr$match, p = 0.7, list = FALSE)
st_training <- st_dating_dr[inTrain,]
st_testing <- st_dating_dr[-inTrain,]


### knn
predicted_match = NULL
error_rate = NULL
for(i in 1:30){
  set.seed(1)
  predicted_match = knn(training, testing, train_match, k=i)
  error_rate[i] = mean(test_match != predicted_match)
}
print(error_rate)
### get the index of that error rate, which is the k
(min_error_rate = min(error_rate))
k <- which(error_rate == min_error_rate)
predicted_match <- knn(training, testing, train_match, k)
confusionMatrix(predicted_match, testing$match, positive = "1") 

## lda
require(MASS)
fitlda <- lda(match ~ ., data = st_training)
table(predict(fitlda)$class, st_training$match)
predlda <- predict(fitlda, st_testing)
confusionMatrix(predlda$class, st_testing$match, positive = "1") 
fitlda_cv <- lda(match ~ ., data = st_dating_dr, CV = TRUE)
confusionMatrix(fitlda_cv$class, st_dating_dr$match, positive = "1") 
## QDA
modqda <- train(match ~ ., data = st_training, method = "qda")
pred_qda <- predict(modqda, st_testing)
confusionMatrix(pred_qda, st_testing$match, positive = "1")

## NB
require(e1071)
require(caret)
require(klaR)
modnb <- train(match ~ ., data = st_training, method = "nb")
pred_nb <- predict(modnb, st_testing)
confusionMatrix(pred_nb, st_testing$match, positive="1")

## Random Forest
require(randomForest)
modrf <- randomForest(match ~ ., st_training, importance=T)
varImpPlot(modrf)
predicted_match <- predict(modrf, st_testing)
table(predicted_match, st_testing$match)
confusionMatrix(predicted_match, st_testing$match, positive="1")

## boosting
require(gbm)
modboost <- train(match ~ ., method = "gbm", data = st_training, verbose = FALSE)
predicted_match <- predict(modboost, st_testing)
table(predicted_match, st_testing$match)
confusionMatrix(predicted_match, st_testing$match, positive="1")

## svm
# linear
require(e1071)
set.seed(12345)
modsvm <- tune(svm, match~., data = st_training, kernel = "linear", 
               ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
modsvm_best <- modsvm$best.model
summary(modsvm_best)
predicted_match <- predict(modsvm_best, st_testing)
table(predicted_match, st_testing$match)
confusionMatrix(predicted_match, st_testing$match, positive="1") 

# radial
set.seed(10000)
modsvm_r <- tune(svm, match ~ ., data = st_training, kernel = "radial", 
                 ranges = list(cost = c(0.1, 1, 10, 100,1000), 
                               gamma = c(0.5,1,2,3,4)))
summary(modsvm_r)
modsvm_r_best <- modsvm_r$best.model
summary(modsvm_r_best)
predicted_match <- predict(modsvm_r_best, st_testing)
table(predicted_match, st_testing$match)
confusionMatrix(predicted_match, st_testing$match, positive = "1")

## logistic
modlogit <- glm(match ~ ., data = st_training, family = "binomial")
summary(modlogit)
predmatch <- predict(modlogit, st_testing, type="response")
predicted_match <- NULL
predicted_match[predmatch >= 0.5] <- 1
predicted_match[predmatch < 0.5] <- 0
table(predicted_match, st_testing$match)
confusionMatrix(predicted_match, st_testing$match, positive="1")

## NNet
require(nnet)
modnn <- nnet(match~., data = st_training, size = 20)
predicted_match <- predict(modnn, st_testing, type="class")
table(predicted_match, st_testing$match)
confusionMatrix(predicted_match, st_testing$match, positive="1")
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
plot(modnn)
