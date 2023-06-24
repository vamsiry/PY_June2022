

df <- read.csv("abc.csv",stringsAsFactors = T) #59,381 observations, 128 variables

dim(df)

head(df)

summary(df)

table(df$target)


sum(colSums(is.na(train))>0) #no.of columnas havig missing values
sum(rowSums(is.na(train))>0)  #no.of rows havig missing values

#checking no.of missing values for each column
data.frame(sort(apply(df, 2, function(x) { sum(is.na(x)) })))

#checking %ge of missing values for each column
pMiss <- function(x){sum(is.na(x))/length(x)*100}
data.frame(sort(apply(train,2,pMiss)))



train[train$Q1==999,"Q1"] <- NA  #Convert a value to missing
mydata$v1[mydata$v1==99] <- NA #Recoding Values to Missing


#need correction
apply(df, 2, function(x) {if(is.numeric(x)) ifelse(is.na(x), mean(x,na.rm=T),x) else x})


df <- na.omit(df) #extract samples without NA 


df2 = train[complete.cases(df),] #extract rows, not having missing values


# Identifying numeric variables
numericData <- df[sapply(df, is.numeric)]

# Calculate correlation matrix
descrCor <- cor(numericData)

#------------------------------------------------------------------------
# pca using caret package
library(caret)
preObj = preProcess(df, method=c("pca"), thresh = 1.0)
newdata = predict(preObj,df)
dim(newdata)


#--------------------------------------------------------
sample_data = sample.split(readingSkills, SplitRatio = 0.8)
train_data <- subset(readingSkills, sample_data == TRUE)
test_data <- subset(readingSkills, sample_data == FALSE)


#-------------------------------------
#KNN
control = trainControl(method = "cv", number = 4)
knn_model_control3 = train(features,data=train_data,method="knn",trControl =  control)



#-----------------------------------------------
#linear regression
fit1 <- lm(y ~ x1 + x2 + x3 + x4, data=mydata)
fit2 <- lm(y ~ x1 + x2)
anova(fit1, fit2)
summary(fit) # show results
coefficients(fit) # model coefficients

library(DAAG)
cv.lm(df=mydata, fit, m=3) # 3 fold cross-validation

library(MASS)
fit <- lm(y~x1+x2+x3,data=mydata)
step <- stepAIC(fit, direction="both")
step$anova # display results 



#---------------------------------------------------------------------
#logistic regression
fit <- glm(F~x1+x2+x3,data=mydata,family=binomial())
summary(fit) # display results
confint(fit) # 95% CI for the coefficients

predict(fit, type="response") # predicted values
residuals(fit, type="deviance") # residuals 



#----------------------------------------------------------
#Decission tree
install.packages("party")
output.tree <- ctree(nativeSpeaker ~ age + shoeSize + score, data = input.dat)

plot(output.tree)

predict_model<-predict(ctree_, test_data)

m_at <- table(test_data$nativeSpeaker, predict_model)

m_at

ac_Test < - sum(diag(table_mat)) / sum(table_mat)


#-----------------------------------------------------------------
library(rpart)
tm <- rpart(quality~., wine_train, method = "class")

library(rpart.plot)
rpart.plot(tm, tweak = 1.6)


#-------------------------------------------------------------


prime_numbers <- function(n) {
  if (n >= 2) {
    x = seq(2, n)
    prime_nums = c()
    for (i in seq(2, n)) {
      if (any(x == i)) {
        prime_nums = c(prime_nums, i)
        x = c(x[(x %% i) != 0], i)
      }
    }
    return(prime_nums)
  }
}


prime_numbers(10)




#-----------------------------------------------------
recursive.factorial <- function(x) {
if (x == 0)    
return (1)
else           
return (x * recursive.factorial(x-1))
}



#----------------------------------------------------------------------

sum_series <- function(vec) {
if(length(vec)<=1)   
 {  return(vec^2)    
}        
  else        
 {  return(vec[1]^2+sum_series(vec[-1]))   
 }
}         


series <- c(1:10)                  
sum_series(series)

#--------------------------------------------
library(data.table)
data$Variable <- as.factor(data$Variable)
newdata <- one_hot(as.data.table(data))


library(caret)

dummy <- dummyVars(" ~ .", data=data)
newdata <- data.frame(predict(dummy, newdata = data)) 
