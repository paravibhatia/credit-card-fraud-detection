install.packages("data.table")
install.packages("caTools")
install.packages("pROC")
install.packages("rpart.plot")
install.packages("neuralnet")
install.packages("devtools")
devtools::install_github("gbm-developers/gbm3")

library(ranger)
library(caret)
library(data.table)
library(pROC)
library(gbm3)


creditcard_data <- read.csv("C:/Users/HP/OneDrive/Documents/creditcard.csv")
#number of rows and columns
dim(creditcard_data)

#showing first and last 6 rows
head(creditcard_data,6)
tail(creditcard_data,6)

#classifies data into fraud credit card and non-fraud credit card
table(creditcard_data$Class)

# Summary statistics for the 'Amount' column to assess transaction distribution,
# including min, max, mean, median, and quartiles. This helps identify trends 
# and potential outliers relevant for fraud detection.
summary(creditcard_data$Amount)

#names of columns
names(creditcard_data)

var(creditcard_data$Amount)
sd(creditcard_data$Amount)

#data manipulation
head(creditcard_data)

#standardizes the Amount column by scaling it to have a mean of 0 and a standard deviation of 1. 
creditcard_data$Amount=scale(creditcard_data$Amount)
#removing 1st column
NewData=creditcard_data[,-c(1)]
head(NewData)

#data modelling
library(caTools)

#helps in random sampling
set.seed(123)

#spliting of data
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)

dim(train_data)
dim(test_data)

#glm function--> generalised linear model, building a model on training data
#family= binomial means binary classification
Logistic_Model = glm(Class ~ ., train_data, family = binomial())
summary(Logistic_Model)
plot(Logistic_Model)

library(pROC)
lr.predict <- predict(Logistic_Model,train_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")

lr.predict <- predict(Logistic_Model, test_data, type = "response")
lr.predict

#thresholding
lr.class <- ifelse(lr.predict > 0.5, 1, 0)

#confusion matrix
confusion_matrix <- confusionMatrix(factor(lr.class), factor(test_data$Class))
cm_table <- as.data.frame(confusion_matrix$table)
ggplot(data = cm_table, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "pink", high = "steelblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix", x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

#dekhng isko rkhna h ya ni
confusion_matrix <- confusionMatrix(factor(lr.class), factor(test_data$Class))
print(confusion_matrix)

library(rpart)
library(rpart.plot)

#decsion tree
decisionTree_model <- rpart(Class ~ . , creditcard_data, method = 'class')
predicted_val <- predict(decisionTree_model, creditcard_data, type = 'class')
probability <- predict(decisionTree_model, creditcard_data, type = 'prob')
rpart.plot(decisionTree_model)

confusion_matrix <- confusionMatrix(factor(predicted_val), factor(creditcard_data$Class))

# Extract the confusion matrix table from the list and convert it to a dataframe

cm_table <- as.data.frame(confusion_matrix$table)

# Plot the confusion matrix
ggplot(data = cm_table, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "steelblue") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix", x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

library(neuralnet)
ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)

predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)

# Confusion Matrix
confusion_matrix <- table(Predicted = resultANN, Actual = test_data$Class)

confusion_matrix1 <- confusionMatrix(factor(resultANN), factor(test_data$Class))

# Extract the table as a dataframe
cm_table <- as.data.frame(confusion_matrix1$table)

# Plot confusion matrix as heatmap
ggplot(data = cm_table, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "orange", high = "steelblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix for ANN Model", x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

library(gbm, quietly=TRUE)

system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)

gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)

plot(model_gbm)

pred_prob <- predict(model_gbm, test_data, n.trees = gbm.iter, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
confusion_matrix <- table(Predicted = pred_class, Actual = test_data$Class)

# Step 2: Convert the confusion matrix into a data frame for ggplot2
confusion_df <- as.data.frame(confusion_matrix)
colnames(confusion_df) <- c("Predicted", "Actual", "Freq")

# Step 3: Plot the confusion matrix
ggplot(data = confusion_df, aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  geom_text(aes(label = Freq), vjust = 1.5, color = "black", size = 5) +
  labs(title = "Confusion Matrix",
       x = "Actual Class",
       y = "Predicted Class") +
  theme_minimal()
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")

print(gbm_auc)
