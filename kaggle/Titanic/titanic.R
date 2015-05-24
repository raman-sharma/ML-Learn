### Set working directory
setwd("~/MyData/R Study/titanic_data")

### Load Data
train <- read.csv("~/MyData/R Study/titanic_data/train.csv")
test <- read.csv("~/MyData/R Study/titanic_data/test.csv")

library(rattle)
library(rpart.plot)
library(RColorBrewer)

### Combine the outputs
test$Survived <- NA
combi <- rbind(train, test)

### Convert factor to String
combi$Name <- as.character(combi$Name)

### R Apply function
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

### Replace character
### sub : first occurance
### gsub : all occurances
combi$Title <- sub(' ', '', combi$Title)
table(combi$Title)
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

### Convert to factor
combi$Title <- factor(combi$Title)

### Add column data values
combi$FamilySize <- combi$SibSp + combi$Parch + 1

### Apply a custom function
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

### Combine two columns
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
table(combi$FamilyID)

### Cleaning data 
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

### Break into parts
train <- combi[1:891,]
test <- combi[892:1309,]

### Decision Tree fit model
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data=train, method="class")

### Replace missing values
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])
summary(combi)
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

### Condition forest example
library(party)
set.seed(415)
train <- combi[1:891,]
test <- combi[892:1309,]
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, test, OOB=TRUE, type = "response")

### Random Forest
#library(randomForest)
#set.seed(415)
#fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, data=train, importance=TRUE, ntree=2000)


### Create Dataframe and write to file
#submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
#write.csv(submit, file = "tut.csv", row.names = FALSE)

### Naive-Bayes Example
library(pROC)
library(e1071)
combi$Survived<-as.numeric(as.character(combi$Survived))
train <- combi[1:800,]
#test <- combi[801:891,]
fit<-naiveBayes(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data = train)
prob<-predict(fit,train, type="class")
prob<-as.numeric(as.character(prob))
nb_test <- data.frame(Survived = train$Survived, prob = prob)
ROC <- roc(Survived~prob, data = nb_test)
plot(ROC)
table(nb_test$prob, nb_test$Survived)

### Logistic Regression Example
library(ISLR)
combi$Survived<-as.numeric(as.character(combi$Survived))
train <- combi[1:891,]
#fit<-glm(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data = train, family=gaussian)
fit<-glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data = train, family=quasipoisson)
summary(fit)
coef(fit)
probs=predict(fit,train,type="response")
nb_test <- data.frame(Survived = train$Survived, prob = probs)
nb_test$pred=rep(0,891)
nb_test$pred[probs>.5]=1
ROC <- roc(Survived~prob, data = nb_test)
plot(ROC)
table(nb_test$pred, nb_test$Survived)


### SVM Example
combi$Survived<-as.numeric(as.character(combi$Survived))
train <- combi[1:891,]
fit=svm(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data=train, kernel="radial",gamma=2, cost=1,decision.values=T)
pred=predict(fit,train,type="class")
pred
nb_test <- data.frame(Survived = as.numeric(as.character(train$Survived)), pred = as.numeric(as.character(pred)))
ROC <- roc(Survived~pred, data = nb_test)
plot(ROC)
table(nb_test$pred, nb_test$Survived)

prf <- function(predAct){
  ## predAct is two col dataframe of pred,act
  preds = predAct[,1]
  trues = predAct[,2]
  xTab <- table(preds, trues)
  clss <- as.character(sort(unique(preds)))
  r <- matrix(NA, ncol = 4, nrow = 1, 
              dimnames = list(c(),c('Acc',
                                    paste("P",clss[1],sep='_'), 
                                    paste("R",clss[1],sep='_'), 
                                    paste("F",clss[1],sep='_'))))
  r2 <- matrix(NA, ncol = 4, nrow = 1, 
              dimnames = list(c(),c('Acc',
                                    paste("P",clss[2],sep='_'), 
                                    paste("R",clss[2],sep='_'), 
                                    paste("F",clss[2],sep='_'))))
  r[1,1] <- sum(xTab[1,1],xTab[2,2])/sum(xTab) # Accuracy
  r[1,2] <- xTab[1,1]/sum(xTab[,1]) # Miss Precision
  r[1,3] <- xTab[1,1]/sum(xTab[1,]) # Miss Recall
  r[1,4] <- (2*r[1,2]*r[1,3])/sum(r[1,2],r[1,3]) # Miss F
  r2[1,1] <- sum(xTab[1,1],xTab[2,2])/sum(xTab) # Accuracy
  r2[1,2] <- xTab[2,2]/sum(xTab[,2]) # Hit Precision
  r2[1,3] <- xTab[2,2]/sum(xTab[2,]) # Hit Recall
  r2[1,4] <- (2*r2[1,2]*r2[1,3])/sum(r2[1,2],r2[1,3]) # Hit F
  r3 = rbind(r,r2)
  r3
}

prf(nb_test)
