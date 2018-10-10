df<-read.csv("F:\\Aegis\\Machine Learning\\Topic 6\\credit.csv")
summary(df)
View(df)
str(df)
head(df)
table(df$default)

## Data Pre-Processing
df$installment_rate=factor(df$installment_rate)
df$residence_history=factor(df$residence_history)
df$existing_credits=factor(df$existing_credits)
df$dependents=factor(df$dependents)
df$default=df$default-1
df$default=factor(df$default)
table(df$default)


library(caTools)
set.seed(1)
x<-sample.split(df$default,SplitRatio = 0.7)
train<-subset(df,x==T)
test<-subset(df,x==F)
colnames(df)

### C50 tree model
library(C50)
model_c<-C5.0(default~.,data = train)
summary(model_c)   ## top 3 are checking_balance, property, savings balance.
# plot(model_c) - getting error of index

result_c<-predict(model_c,test)
table(result_c,test$default)


## 2 more ways of a more descriptive confusion matrix :-
# library(caret)
# confusionMatrix(result_c,test$default)

# library(gmodels)
# CrossTable(result_c,test$default)

## Improving The Performance of c5.0
## The c5.0 has a parameter trials which is a sort of boosting method for improving
## the performance of the model, the value for it specified the number of tree models used
## for building it.

model_c1<-C5.0(default~.,data = train,trials=30)
summary(model_c1)
result_c1<-predict(model_c1,test)
table("predicted"=result_c1,"actual"= test$default)

## From varying the trials values, for higher values = 50, it was giving overfitting,i.e.,
## Attribute Usage increased drastically to 100% for many variables and others were around
## 90-95% usage, and also the confusion matrix table showed high misclassification error,
## giving high false negatives. So we choose a trade-off value of 30.




### rpart tree model
library(rpart)
library(rpart.plot)
model_r<-rpart(default~., data = train, method = "class")  ## method indicates classification problem.
summary(model_r)  ## top 3 are checking balance, purpose, months loan duration and amt.
printcp(model_r)  ##This gives the cross-validation error: CP means Complexity Parameter,
        ##indicates the amount by which the error drops due to a split at a particular node. 
        ## By default, tree stops building at an improvement of 0.01.

plotcp(model_r)  ## plots cross-validation results.,i.e, xerror vs CP (need to select that value of cp, having minimum xerror)
prp(model_r)

result_r<-predict(model_r,test,type = "class")
table("predicted"=result_r,"actual"=test$default)

## Improving The Performance of Rpart.
cp_optimum<-model_r$cptable[which.min(model_r$cptable[,"xerror"]),"CP"] ## selecting cp value for the min xerror.
                                    ## which() function is used to return the index position.
ptree<-prune(model_r, cp = cp_optimum)
printcp(ptree)
prp(ptree)  ## New pruned tree.
result_r1<-predict(ptree,test,type = "class")
table("predicted"=result_r1,"actual"=test$default)


### Random Forest model
library(randomForest)
model_rf<-randomForest(default~.,data = train) ## default ntree=500 and importance = FALSE
model_rf
plot(model_rf)
model_rf$confusion  ## confusion matrix with itself.
model_rf$importance
## From the Decreasing Gini index obtained by $importance, top 3 variables are 
## amount,checking balance, months loan duration and age.

result_rf<-predict(model_rf,test)
table("predicted"=result_rf,"actual"=test$default)



### Improving the performance of Random Forest

tuneRF(train[,-21],train[,21]) ## from this plot we see that for mtry=4 (default value taken by model),
## we get the minimum error, and remains the same beyond that, so won't change mtry. 

## So we would change maxnodes, nodesize and ntree
# maxnodes is the max depth for a single tree. More depth, more is the chance of overfitting.
# nodesize is the minimum size of terminal nodes(leaf), the less size, more is the overfitting.
model_rf1<-randomForest(default~.,data = train,ntree= 500,maxnodes=80,nodesize=8)
model_rf1$confusion

result_rf1<-predict(model_rf1,test)
table("predicted"=result_rf1,"actual"=test$default)




### With Missing data
i<-1
train_na<-train
for(i in 1:length(train_na))
{
 train_na[sample(seq(train_na[,i]),50),i]=NA
 
}
summary(train_na)
View(train_na)


## c50
model_c1_na<-C5.0(default~.,data = train_na,trials=30)
summary(model_c1_na)
result_c1_na<-predict(model_c1_na,test)
table("predicted"=result_c1_na,"actual"= test$default)
## We see a fairly similar confusion matrix like the original one without the NA, hence the 
## algorithm is taking care of imputing the Na values internally.

## rpart
model_r_na<-rpart(default~., data = train_na, method = "class") 
summary(model_r_na)
printcp(model_r_na)  

plotcp(model_r_na)  
prp(model_r_na)


cp_optimum<-model_r_na$cptable[which.min(model_r_na$cptable[,"xerror"]),"CP"] 
ptree_na<-prune(model_r_na, cp = cp_optimum)
printcp(ptree_na)
prp(ptree_na)  
result_r1_na<-predict(ptree_na,test,type = "class")
table("predicted"=result_r1_na,"actual"=test$default)
## We see a fairly similar confusion matrix like the original one without the NA


## Random forest
model_rf1_na<-randomForest(default~.,data = train_na,na.action = na.omit)
model_rf1_na$confusion

result_rf1_na<-predict(model_rf1_na,test)
table("predicted"=result_rf1_na,"actual"=test$default)
## Here the confusion matrix has more of false negative error( predicted 0, when it is actually 1)
## since in random forest we are omitting the na values, hence information is lost.
