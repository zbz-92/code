# PRELIMINARY ----
## Set the working directory
setwd("C:/Users/00131045/Documents/FutureReady/Challenge Cards")

## Install packages
library(data.table)
library(randomForestSRC)
library(ROSE)
library(mlr)
library(tidyverse)
library(ggplot2)
library(parallel)
library(parallelMap)
library(rpart.plot)

# AUDIT ----
## Clean data
## 1.Data collection. Dataset has been reduced from 99 variables to 49 variables after assessing correlations.
response <- fread("Dataset Cards clean.csv", header = TRUE, stringsAsFactors = TRUE)
responseTib <- as_tibble(response)

## 2.Segregate factor and integer features
toFactor <- c("take_up_flag", "Marital_Status_Desc", "Gender", "Education_Desc", "Race_Desc",
              "F_CASA", "F_FD", "F_Inv", "F_Ins", "F_HP", "F_CC", 
              "m2u_flag", "PMEB_FLAG", "Kid_flag", "CHANNEL_PREF", "CREDIT_RISK", 
              "location", "region", "PROFIT_PERCENTILE", "INV_RISK", "GOLF_SEGMENT", "charge_card_flag", 
              "transactor_revolver_flag", "Segment", "redefine_segment", "Income_Group", 
              "bt_flag", "ct_flag", "ezcash_flag", "cash_adv_flag", "Occ_Group")

toInt <- c("CASA_Amt", "FD_Amt", "Inv_amt", "Ins_Amt", "HP_amt", "CC_amt", 
           "bank_vintage", "NO_OF_ACCT", "ANNUALISED_CV", "CC_Limit", "Ratio_Acc_Open_Dt", "diff_account_open", 
           "total_p2m_TFA", "CC_Bal_P1M", "withdrawal_txn_1mths","ct_amt_P6m", "ezcash_amt_P6m", "cash_adv_amt_P6m")

responseTib <- responseTib %>% 
  mutate_at(toFactor, as.factor) %>% 
  mutate_at(toInt, as.integer)

## 3. Multicollinearity test
model.logistic.fit <- glm(take_up_flag ~ .,family = "binomial", data = responseTib)
1-model.logistic.fit$deviance/model.logistic.fit$null.deviance            # assess overall effect size with McFadden's Pseudo Rsquared
1-pchisq(model.logistic.fit$null.deviance - model.logistic.fit$deviance,  # chiSquared test (tiny => relationship not due to luck)
         df = length(model.logistic.fit$coefficients)-1) 

## 4. Check if data is balanced
table(responseTib$take_up_flag)
prop.table(table(responseTib$take_up_flag))

## 5. Check missing data
sum(is.na(responseTib))

## 6. Rebalance if necessary (undersampling)
# Undersampling
response.under <- ovun.sample(take_up_flag ~ ., 
                              data = responseTib, 
                              method = "under", 
                              seed = 42)$data
table(response.under$take_up_flag)
prop.table(table(response.under$take_up_flag))

response2 <- select(response.under, -c(Occ_Group, F_CC, charge_card_flag,
                                   PMEB_FLAG, cash_adv_flag, ct_flag,
                                   Gender, F_FD, F_Inv, cash_adv_amt_P6m,
                                   bt_flag, INV_RISK, F_Ins, ct_amt_P6m, 
                                   GOLF_SEGMENT))
response.num <- select_if(response2, is.numeric)
response.fctr <- select_if(response2, is.factor)

# PLOT ----
ggplot(response2,  aes(Education_Desc, Race_Desc, col=take_up_flag))+
  geom_point()

# Create subplots for each categorical variable
responseUntidy.discrete <- gather(response.fctr, key = "Variable", value = "Value",-take_up_flag)

responseUntidy.discrete %>%
  filter(Variable == colnames(response.fctr)[2:length(response.fctr)]) %>%
  ggplot(aes(Value, fill = take_up_flag)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_bar(position = "fill")+
  theme_bw()

# Create subplots for each continuous variable
response.num <- response.num %>% 
  add_column(response2$take_up_flag, .before = 1) %>% 
  rename("take_up_flag" = "response2$take_up_flag")

responseUntidy.num <- gather(response.num, key = "Variable", value = "Value",-take_up_flag)
responseUntidy.num %>%
  filter(Variable == colnames(response.num)[2:length(response.num)]) %>%
  ggplot(aes(take_up_flag, as.numeric(Value))) +
  facet_wrap(~ Variable, scales = "free_y") +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75))+
  theme_bw()

# DT ----
# Create task
takeupTask <- makeClassifTask(data = response2, 
                              target = "take_up_flag", 
                              positive = "1")
takeupTaskFiltered <- filterFeatures(takeupTask, method = "rf.importance", abs = 6)
takeupTask <- takeupTaskFiltered

# Create learner
tree <- makeLearner("classif.rpart", predict.type = "response")

# Define hyperparameter space for tuning
treeParamSpace <- makeParamSet(
  makeIntegerParam("minsplit", lower = 2, upper = 20),   #minimum number of cases needed to split a node
  makeIntegerParam("minbucket", lower = 2, upper = 10),  
  makeNumericParam("cp", lower = 0.01, upper = 0.1),     #complexity parameter ~ performance
  makeIntegerParam("maxdepth", lower = 3, upper = 10)    #minimum number of cases in a leaf
)

# Define random search
randSearch.tree <- makeTuneControlRandom(maxit=200)   #not exhaustive unlike grid search
cvForTuning.tree <- makeResampleDesc("RepCV", folds=5, reps=3)   #5-vold cross validation, repeated 3 times

# Perform hyperparameter tuning
set.seed(42)
parallelStartSocket(cpus = detectCores())
tunedTreePars <- tuneParams(tree, 
                            task = takeupTask,
                            resampling = cvForTuning.tree,
                            par.set = treeParamSpace,
                            control = randSearch.tree)
parallelStop()
tunedTreePars

# Training the final tuned model
tunedTree <- setHyperPars(tree, par.vals = tunedTreePars$x)
tunedTreeModel <- train(tunedTree, takeupTask)

# Plot the decision tree
treeModelData <- getLearnerModel(tunedTreeModel)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type = 2)

# Exploring the model
printcp(treeModelData, digits = 3)
summary(treeModelData)
dtImp <- getFeatureImportance(tunedTreeModel)
dtImpTib <- tibble(Variable = colnames(dtImp$res),
                   Score = dtImp$res)

# Cross-validating model-building process
outer.tree <- makeResampleDesc("RepCV", folds=5, reps=3)
treeWrapper <- makeTuneWrapper("classif.rpart", 
                               resampling = cvForTuning.tree,
                               par.set = treeParamSpace,
                               control = randSearch.tree)
parallelStartSocket(cpus = detectCores())
cvWithTuning.tree <- resample(treeWrapper, takeupTask, resampling = outer.tree)
parallelStop()

cvWithTuning.tree # Extract cross-validation result

# ROC 
calculateROCMeasures(cvWithTuning.tree$pred)

treeProb <- makeLearner("classif.rpart", predict.type = "prob")
tunedTreeProb <- setHyperPars(treeProb, par.vals = tunedTreePars$x)
tunedTreeProbModel <- train(tunedTreeProb, takeupTask)
predTree <- predict(tunedTreeProbModel, takeupTask)
performance(predTree, measures=auc)

# RF ----
# Create learner
forest <- makeLearner("classif.randomForest")

# Define hyperparameter space for tuning
forestParamSpace <- makeParamSet(
  makeIntegerParam("ntree", lower = 50, upper = 300),
  makeIntegerParam("mtry", lower = 6, upper = 12),
  makeIntegerParam("nodesize", lower = 1, upper = 5),
  makeIntegerParam("maxnodes", lower = 5, upper = 20))

# Define random search
randSearch.forest <- makeTuneControlRandom(maxit = 100)
cvForTuning.forest <- makeResampleDesc("CV", iters = 5)

# Tuning
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = takeupTask,
                              resampling = cvForTuning.forest,
                              par.set = forestParamSpace,
                              control = randSearch.forest)
parallelStop()
tunedForestPars 

# Train the final tuned model
tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)
tunedForestModel <- train(tunedForest, takeupTask)

# Plot oob error
forestModelData <- getLearnerModel(tunedForestModel)
plot(forestModelData)
species <- colnames(forestModelData$err.rate)
legend("topright", species,
       col = 1:length(species),
       lty = 1:length(species))

rfImp <- getFeatureImportance(tunedForestModel)
rfImp$res

# Cross-validating
outer <- makeResampleDesc("CV", iters = 5)
forestWrapper <- makeTuneWrapper("classif.randomForest", 
                                 resampling = cvForTuning.forest,
                                 par.set = forestParamSpace,
                                 control = randSearch.forest)
parallelStartSocket(cpus = detectCores())
cvWithTuning <- resample(forestWrapper, takeupTask, resampling = outer) 
parallelStop()
cvWithTuning

# ROC
calculateROCMeasures(cvWithTuning$pred)

forestProb <- makeLearner("classif.randomForest", predict.type = "prob")
tunedForestProb <- setHyperPars(forestProb, par.vals = tunedForestPars$x)
tunedForestProbModel <- train(tunedForestProb, takeupTask)
predForest <- predict(tunedForestProbModel, takeupTask)
performance(predForest, measures=auc)

# XGBOOST ----
xgb <- makeLearner("classif.xgboost", par.vals = list(objective = "multi:softmax"))

# Convert all column except target as numeric
responseXgb <- response2 %>% 
  mutate_at(.vars = vars(-take_up_flag), .funs = as.numeric) 
xgbTask <- makeClassifTask(data = responseXgb, 
                           target = "take_up_flag",
                           positive = "1")

# Tuning XGBoost hyperparameters
xgbParamSpace <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 1),
  makeNumericParam("gamma", lower = 0, upper = 5),
  makeIntegerParam("max_depth", lower = 1, 5),
  makeNumericParam("min_child_weight", lower = 1, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeIntegerParam("nrounds", lower = 20, upper = 100),
  makeDiscreteParam("eval_metric", values = c("merror", "mlogloss")))

randSearch.xgb <- makeTuneControlRandom(maxit = 300)
cvForTuning.xgb <- makeResampleDesc("CV", iters=3)

parallelStartSocket(cpus = detectCores())
tunedXgbPars <- tuneParams(xgb, 
                           task = xgbTask,
                           resampling = cvForTuning.xgb,
                           par.set = xgbParamSpace,
                           control = randSearch.xgb)
parallelStop()
tunedXgbPars

# Train final tuned model
tunedXgb <- setHyperPars(xgb, par.vals = tunedXgbPars$x)
tunedXgbModel <- train(tunedXgb, xgbTask)

xgbModelData <- getLearnerModel(tunedXgbModel)
ggplot(xgbModelData$evaluation_log, aes(iter, train_mlogloss)) +
  geom_line() +
  geom_point()

xgboost::xgb.plot.tree(model = xgbModelData, trees = 1)
xgboost::xgb.plot.multi.trees(xgbModelData)

xgbImp <- getFeatureImportance(tunedXgbModel)
xgbImp <- t(xgbImp$res)

rfImp <- getFeatureImportance(tunedForestModel)
rfImp <- t(rfImp$res)

treeImp <- getFeatureImportance(tunedTreeModel)
treeImp <- t(treeImp$res)


# Cross-validation
outer.xgb <- makeResampleDesc("CV", iters=3)
xgbWrapper <- makeTuneWrapper(xgb,
                              resampling = cvForTuning.xgb,
                              par.set = xgbParamSpace,
                              control = randSearch.xgb)
parallelStartSocket(cpus = detectCores())
cvWithTuning.xgb <- resample(xgbWrapper, xgbTask, resampling = outer.xgb)
parallelStop()
cvWithTuning.xgb

#ROC
calculateROCMeasures(cvWithTuning.xgb$pred)
xgb <- makeLearner("classif.xgboost", 
                   predict.type = "prob",
                   par.vals = list(objective = "multi:softprob"))
tunedXgboostProb <- setHyperPars(xgb, par.vals = tunedXgbPars$x)
tunedXgboostProbModel <- train(tunedXgboostProb, xgbTask)
predXgboost <- predict(tunedXgboostProbModel, xgbTask)
performance(predXgboost, measures=auc)
calculateROCMeasures(predXgboost)
cvWithTuning.xgb$pred

# Ensemble ----
VarImp <- tibble(Feature = colnames(getFeatureImportance(tunedXgbModel)$res), 
                 Decision_Tree = treeImp,
                 Random_Forest = rfImp, 
                 XGBoost = xgbImp)
scaledVarImp <- normalizeFeatures(VarImp, method = "range")

ggplot(data = scaledVarImp, aes(y = Decision_Tree , x = Feature, fill = Feature)) + 
  geom_point(color="green", shape=15, size=3) + 
  geom_point(data = scaledVarImp, aes(y = Random_Forest , x = Feature, fill = Feature),
             color="red", shape=15, size=3) + 
  geom_point(data = scaledVarImp, aes(y = XGBoost , x = Feature, fill = Feature),
             color="orange", shape=15, size=3) + 
  coord_flip() + labs(title= 'Variable importance plot') +
  theme(legend.position = "none")