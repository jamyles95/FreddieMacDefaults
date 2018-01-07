rm(list=ls())
library(randomForest)
library(mlr)
library(discSurv) ##https://cran.r-project.org/web/packages/discSurv/discSurv.pdf
library(data.table)
library(reshape)
library(vcdExtra)
library(plyr)
library(dplyr)
library(tidyverse)
library(haven)
library(tidyverse)
library(lattice)
library(gmodels)
library(vcd)
library(exact2x2)
library(caret)
library(xgboost)
library(Matrix)


setwd("C:\\Users\\Jabari\\Desktop\\Extra Projects\\Defaults\\historical_data1_Q12008")


####### Creates vector of variable types for the Origination dataset #########
origclass <- c('integer','integer','character', 'integer', 'character', 'real', 'integer',
               'character','real','integer','integer','integer','real','character','character','character','character',
               'character','character','character','character', 'integer', 'integer','character','character' ,'character' )


######## Reads in origination dataset, with pipe separator, without a header, and sets the variable types ##########
origfile_Q12008 <- read.table("historical_data1_Q12008.txt", sep="|", header=FALSE, colClasses=origclass)


######## Sets the name of each variable in the Origination dataset #########
names(origfile_Q12008)=c('fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts','cltv'
                         ,'dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', 'prop_type','zipcode','id_loan','loan_purpose',
                         'orig_loan_term','cnt_borr','seller_name','servicer_name', 'flag_sc') 


####### Creates vector of variable types for the Performance dataset #########
svcgclass <- c('character','integer','real','character', 'integer','integer','character','character',
               'character','integer','real','real','integer', 'integer', 'character','integer','integer',
               'integer','integer','integer','integer','real','real')


######## Reads in Performance dataset, with pipe separator, without a header, and sets the variable types ##########
svcgfile_Q12008 <- read.table("historical_data1_time_Q12008.txt", sep="|", header=FALSE, colClasses=svcgclass)


######## Sets the name of each variable in the Performance dataset #########
names(svcgfile_Q12008)=c('id_loan','svcg_cycle','current_upb','delq_sts','loan_age','mths_remng', 'repch_flag','flag_mod', 'cd_zero_bal',
                         'dt_zero_bal','current_int_rt','non_int_brng_upb','dt_lst_pi','mi_recoveries',
                         'net_sale_proceeds','non_mi_recoveries','expenses', 'legal_costs',
                         'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost')


## Takes these out below: channel, repch_flag, cd_zero_bal, st, prod_type,orig_upb, loan_purpose
######## Creates vector for variables of interest for the Origination dataset ########
origvars <- c('fico','flag_fthb','mi_pct','cnt_units','occpy_sts','dti','ltv','int_rt', 'prop_type','id_loan',
              'orig_loan_term','cnt_borr')


######## Creates vector for variables of interest for the Performance dataset ########
svcgvars <- c('id_loan','delq_sts','loan_age','mths_remng','flag_mod',
              'current_int_rt')


######## Subsets originally datasets to merge ########
origfile <- origfile_Q12008 %>% select(one_of(origvars))

svcgfile <- svcgfile_Q12008 %>% select(one_of(svcgvars))


####### Removes non-subsetted datasets to free up memory ######
remove(origfile_Q12008)
remove(svcgfile_Q12008)

###### Increase memory to allow merging ##########
memory.limit(size = 200000)

####### Index each file by the id_loan variable to cut down on calculation time (not sure, but I think R does the cartisian product) #########
origfile <- origfile %>%
  group_by(id_loan)

svcgfile <- svcgfile %>%
  group_by(id_loan)

###### Performs one (origfile) to many (svcgfile) merge #######
combinedData <- data.table(origfile, key ="id_loan")[data.table(svcgfile, key = "id_loan"), allow.cartesian = FALSE]

save(combinedData, file="combinedData.csv", compress = F)



#### Free up memory to export to CSV ######
remove(origfile)
remove(svcgfile)

#### Exports to CSV and names it combinedData.csv #######
write.csv(combinedData, "combinedData.csv")

set.seed(3456)

index <- sample(1:nrow(combinedData), nrow(combinedData)*0.2)

combinedDataSamp <- combinedData[index,]

rm(combinedData)

#Takes out all observations where delinquency status is equal to REO Acquisition or Unknown
combinedDataSamp <- filter(combinedDataSamp, delq_sts != "R" | delq_sts != "XX" ) 


combinedDataSamp$Delq30 <- rep(0, nrow(combinedDataSamp))


combinedDataSamp$Delq30 <- if_else(combinedDataSamp$delq_sts >= 1, 1, 0)

#6.99826 percent defaulted
mean(combinedDataSamp$Delq30)

###### DESCRIPTIVE STATS #######

# 1 = survived, 0 = died
# when you make something a factor, the first level will be your reference level
combinedDataSamp$Delq60two <- factor(combinedDataSamp$Delq60, levels = c(0,1),
                                     labels = c("paid", "defaulted"))
# 1, 2, 3 = 1st, 2nd, 3rd class, respectively
titanic$Class2 <- factor(titanic$Class, levels = c(1,2,3),
                         labels = c("1st", "2nd", "3rd"))
titanic$Gender2 <- factor(titanic$Gender, levels = c("female", "male"),
                          labels = c("female", "male"))



####################
######ANALYSIS######
#####STARTS HERE####
####################
glimpse(combinedDataSamp)

set.seed(502)

##### CREATE TRAINING AND TEST SET ######

split <- createDataPartition(y = combinedDataSamp$Delq30, p = 0.7, list = F)

train <- combinedDataSamp[split,]

test <- combinedDataSamp[-split,]

glimpse(train)


#Creates binary variable for first time home buyers
train$fthb <- ifelse(train$flag_fthb == "Y", 1,ifelse(train$flag_fthb == "N", 0, NA))

train$mod  <- ifelse(train$flag_mod == "Y", 1,0)

train$OwnerOcc <- if_else(train$occpy_sts == "O", 1,0, missing = 0)
train$InvOcc <- if_else(train$occpy_sts == "I", 1, 0, missing = 0)
train$SecHomeOcc <- if_else(train$occpy_sts == "S", 1, 0, missing = 0)
train$UnkOcc <- if_else(is.na(train$occpy_sts), 1, 0, missing = 0)

train$Condo <- if_else(train$prop_type == "CO", 1, 0, 0)
train$Leasehold <- if_else(train$prop_type == "LH", 1, 0, 0)
train$PUD <- if_else(train$prop_type == "MH", 1, 0, 0)
train$SF <- if_else(train$prop_type == "SF", 1, 0, 0)
train$CP <- if_else(train$prop_type == "CP", 1, 0, 0)
train$UnkType <- if_else(is.na(train$prop_type), 1, 0)


train <- subset(train, select = -c(delq_sts, id_loan, flag_mod, occpy_sts, prop_type, flag_fthb))

glimpse(train)

## Test Data Transformation Starts Here
test$fthb <- ifelse(test$flag_fthb == "Y", 1,ifelse(test$flag_fthb == "N", 0, NA))

test$mod  <- ifelse(test$flag_mod == "Y", 1,0)

test$OwnerOcc <- if_else(test$occpy_sts == "O", 1,0, missing = 0)
test$InvOcc <- if_else(test$occpy_sts == "I", 1, 0, missing = 0)
test$SecHomeOcc <- if_else(test$occpy_sts == "S", 1, 0, missing = 0)
test$UnkOcc <- if_else(is.na(test$occpy_sts), 1, 0, missing = 0)

test$Condo <- if_else(test$prop_type == "CO", 1, 0, 0)
test$Leasehold <- if_else(test$prop_type == "LH", 1, 0, 0)
test$PUD <- if_else(test$prop_type == "MH", 1, 0, 0)
test$SF <- if_else(test$prop_type == "SF", 1, 0, 0)
test$CP <- if_else(test$prop_type == "CP", 1, 0, 0)
test$UnkType <- if_else(is.na(test$prop_type), 1, 0)

glimpse(test)

#Removes original varibles reformatted above
test <- subset(test, select = -c(delq_sts, id_loan, flag_mod, occpy_sts, prop_type, flag_fthb))




test$delq_sts <- as.numeric(test$delq_sts)

glimpse(train)

remove(combinedDataSamp)
remove(split)

##############
####Random####
####Forest####
##############

rf <- randomForest(Delq30 ~ ., data=train, ntree=25, na.action = na.exclude)

#############
###XGBoost###
#############

sparse_train <- sparse.model.matrix(Delq30 ~ . - Delq30, data = train)
sparse_valid <- sparse.model.matrix(Delq30 ~ . - Delq30, data = test)

train_label <- train$Delq30

train_no_y <- train[c(-12)]

test_no_y <- test[c(-12)]
  
glimpse(train_no_y)



class(sparse_train)

xgb <- xgboost(data = as.matrix(train_no_y),
               label = as.matrix(train$Delq30),
               eta = 0.05,
               max_depth = 30,
               gamma = 0,
               nround = 100,
               subsample =  0.75,
               colsample_bytree = 0.75,
               num_class = 1,
               objective = "binary:logistic",
               nthread = 3,
               eval_metric = 'error',
               eval_metric = 'logloss',
               verbose = 0
)

ptrain <- predict(xgb, as.matrix(train_no_y))
pvalid <- predict(xgb, as.matrix(test_no_y))
table(ptrain, train$Delq30)

tprediction <- as.numeric(ptrain > 0.5)
vprediction <- as.numeric(pvalid > 0.5)

#Variable importance of XGBoost
xgb.importance(colnames(as.matrix(train_no_y)), model = xgb)



errt <- mean(tprediction != train$Delq30)
errv <- mean(vprediction != test$Delq30)

FalsePos <- mean(vprediction == 1 & test$Delq30 == 0)
FalsePosCnt <- FalsePos * nrow(test)
print(paste("The False Positive Rate is ", FalsePos*100,"%"))
print(paste("The number of False Positives is ", FalsePosCnt))

FalseNeg <- mean(vprediction == 0 & test$Delq30 == 1)
FalseNegCnt <- FalseNeg * nrow(test)
print(paste("The False Negative Rate is ", FalseNeg*100,"%"))
print(paste("The number of False Negatives is ", FalseNegCnt))

TruePos <- mean(vprediction == 1 & test$Delq30 == 1)
TruePosCnt <- TruePos * nrow(test)
print(paste("The True Positive Rate is", TruePos*100,"%"))
print(paste("The number of True Positives is", TruePosCnt))

TrueNeg <- mean(vprediction == 0 & test$Delq30 == 0)
TrueNegCnt <- TrueNeg * nrow(test)
print(paste("The True Negative Rate is", TrueNeg*100,"%"))
print(paste("The number of True Negatives is", TrueNegCnt))

print(paste("test-error=", errt))
print(paste("validation-error=", errv))

mean(test$Delq30 == 1)

mean(train$Delq30 == 1)

#Saves the model we just created to our working directory. This model can now be reused to predict other defaults
save(xgb, file = "XgbDefault1.rda")

## To resuse this model 
load("XgbDefault1.rda")

predict(xgb, newdata = newdf)

glimpse(train)
?sample
?dataLong
# Convert to long format
CombinedLong <- dataLong (dataSet=combinedData, timeColumn="loan_age", censColumn="censor1")
head(UnempLong)
# Estimate binomial model with logit link
Fit <- glm(formula=y ~ timeInt + age + logwage, data=UnempLong, family=binomial())
# Estimate discrete survival function given age, logwage of first person
hazard <- predict(Fit, newdata=subset(UnempLong, obj==1), type="response")
# Estimate marginal probabilities given age, logwage of first person
MarginalProbCondX <- estMargProb (hazard)
MarginalProbCondX
sum(MarginalProbCondX)==1 # TRUE: Marginal probabilities must sum to 1!

