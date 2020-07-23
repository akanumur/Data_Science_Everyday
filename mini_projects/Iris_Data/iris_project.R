#Installing required packages
install.packages("caret")
# To load the library
library(caret)

# To load the data
mydata <- iris

# Exploring the data
dim(mydata)#dimensions of dataset
sapply(mydata, class) # list types for each attribute
# take a peek at the first 5 rows of the data
head(mydata)
# list the levels for the class
levels(mydata$Species)

# summarize the class distribution
percentage <- prop.table(table(mydata$Species)) * 100
cbind(freq=table(mydata$Species), percentage=percentage)
# summarize the class distribution
percentage <- prop.table(table(mydata$Species)) * 100
# summarize attribute distributions
summary(mydata)
#Visualizing the data
# split input and output
x <- mydata[,1:4]
y <- mydata[,5]
# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(mydata)[i])
}


# barplot for class breakdown
plot(y)

install.packages("ellipse")
library(ellipse)
# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")


# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# Split data into train and test
n = nrow(mydata) # n will be ther number of obs. in data
trainIndex = sample(1:n, 
                    size = round(0.7*n), 
                    replace=FALSE) # We create an index for 70% of obs. by random
train_data = mydata[trainIndex,] # We use the index to create training data
test_data = mydata[-trainIndex,] # We take the remaining 30% as the testing data
summary(train_data)
summary(test_data)

#Model Buliding
# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=train_data, method="lda")
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=train_data, method="rpart")
# kNN
set.seed(7)
fit.knn <- train(Species~., data=train_data, method="knn")
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=train_data, method="svmRadial")
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=train_data, method="rf")

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)


# compare accuracy of models
dotplot(results)


# summarize Best Model
print(fit.lda)


# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, test_data)
confusionMatrix(predictions, test_data$Species)
