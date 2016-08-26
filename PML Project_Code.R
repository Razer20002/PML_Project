#Download data

pml_training <- read.csv("pml-training.csv", header=TRUE)
pml_testing <- read.csv("pml-testing.csv",header=TRUE)


#Clean data

library(e1071)
library(caret)
library(AppliedPredictiveModeling)
set.seed(12345)

#Take the records we are interested in.

pml_train2 <- pml_training[pml_training$new_window == 'yes',]

pml_train2$classe <- as.factor(pml_train2$classe)

pml_train3 <- subset(pml_train2, select= c(classe,user_name,X,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window
                                           ,roll_belt,pitch_belt,yaw_belt,gyros_belt_x,gyros_belt_y,gyros_belt_z,accel_belt_x,accel_belt_y,accel_belt_z,magnet_belt_x,magnet_belt_y,magnet_belt_z
                                           ,roll_arm,pitch_arm,yaw_arm, gyros_arm_x,gyros_arm_y,gyros_arm_z,accel_arm_x,accel_arm_y,accel_arm_z,magnet_arm_x,magnet_arm_y,magnet_arm_z
                                           , roll_dumbbell,pitch_dumbbell,yaw_dumbbell, gyros_dumbbell_x,gyros_dumbbell_y,gyros_dumbbell_z,accel_dumbbell_x,accel_dumbbell_y,accel_dumbbell_z,magnet_dumbbell_x,magnet_dumbbell_y,magnet_dumbbell_z
                                           , roll_forearm,pitch_forearm,yaw_forearm, gyros_forearm_x,gyros_forearm_y,gyros_forearm_z,accel_forearm_x,accel_forearm_y,accel_forearm_z,magnet_forearm_x,magnet_forearm_y,magnet_forearm_z
)
)

pml_train4 <- pml_train3[,-c(2:8)]

#Create principle components
PCA_Base <- subset(pml_train4, select= -c(classe))
PCA <- preProcess(PCA_Base,method="pca",thresh = 0.9)

trainingPCA <- predict(PCA,pml_train4)


#Create models
random_forest <- train(classe ~ . ,method="rf", data=trainingPCA)
print(random_forest)
random_forest$finalModel

#There is an error rate of 29.31% in the train set.
#To increase the accuracy of the random forest I will add a gradient boosting model and combine them.

gbm_mod <- train(classe~.,method="gbm",data=trainingPCA)

rf_predict <- predict(random_forest,trainingPCA)
gbm_predict <- predict(gbm_mod,trainingPCA)

#Combine models

combo_data <- data.frame(rf_predict,gbm_predict, classe=trainingPCA$classe)
combo_mod <- train(classe~., method="rf", data=combo_data)

print(combo_mod)
combo_mod$finalModel
varImp(combo_mod)

#The gmb model is more important and contributing the most to the combination

combo_predict <- predict(combo_mod,combo_data)
combo_data2 <- data.frame(combo_data,combo_predict)

#Cross Validate
CrossValid <- rfcv(trainingPCA, combo_predict, cv.fold=20)
with(CrossValid, plot(n.var, error.cv, log="x", type="o", lwd=2))

#Test on test data
testing_PCA <- predict(PCA,pml_testing)
testing_PCA2 <- testing_PCA[,-c(1:111)]
#testing_PCA2 <- pml_testing
rf_predict <- predict(random_forest,testing_PCA2)
gbm_predict <- predict(gbm_mod,testing_PCA2)

combo_data_test <- data.frame(rf_predict,gbm_predict)

test_restults <- predict(combo_mod,combo_data_test)
```