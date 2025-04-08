library(caret)
library(tidyverse)
library(missForest)
library(missForestPredict)
library(entropy)
source('utils.R')
set.seed(123) # for reproducibility



##### 01. Load the data #####
load('../objects/features_labels.RData')



##### 02. Set aside static data, discretize dynamic data + calculate time series features + pivot wide #####
# static 
static <- features %>% 
  filter(str_detect(Parameter, 'static')) %>% 
  group_by(PATIENT_ID, Parameter, Time) %>% 
  mutate(Value = mean(Value)) %>% 
  as.data.frame() %>% 
  select(-Time, -HR) %>% 
  filter(Parameter != 'static_RecordID') %>% 
  unique(.) %>% 
  pivot_wider(
    id_cols = PATIENT_ID,
    names_from = Parameter, 
    values_from = Value
  ) %>% 
  mutate(
    static_SurgicalICU = ifelse(static_ICUType == 4, 1, 0),
    static_MedicalICU = ifelse(static_ICUType == 3, 1, 0),
    static_CardiacUnit = ifelse(static_ICUType == 2, 1, 0)
    ) %>% 
  select(-static_ICUType)


# discretize dynamic data 
dynamic_discrete <- features %>% 
  mutate(Period = ifelse(HR <= 24, 0, 1)) %>%  # day long periods
  as.data.frame() %>% 
  select(-Time, -HR) %>% 
  filter(str_detect(Parameter, 'dynamic')) %>% 
  filter(!str_detect(Parameter, 'Cholesterol')) %>% # remove cholesterol 
  filter(!str_detect(Parameter, 'Troponin')) %>% # remove Troponin 
  select(PATIENT_ID, Period, everything())

# calculate time series features
ts_means <- dynamic_discrete %>% 
  group_by(PATIENT_ID, Period, Parameter) %>% 
  mutate(Value = mean(Value, na.rm = TRUE)) %>% 
  unique(.) %>% 
  Pivot.Wide(.) %>% 
  rename_with(~paste0("mean_", .), -PATIENT_ID)  

ts_medians <- dynamic_discrete %>% 
  group_by(PATIENT_ID, Period, Parameter) %>% 
  mutate(Value = median(Value, na.rm = TRUE)) %>% 
  unique(.) %>% 
  Pivot.Wide(.) %>% 
  rename_with(~paste0("med_", .), -PATIENT_ID)  

ts_sds <- dynamic_discrete %>% 
  group_by(PATIENT_ID, Period, Parameter) %>% 
  mutate(Value = sd(Value, na.rm=TRUE)) %>% 
  unique(.) %>% 
  Pivot.Wide(.) %>% 
  rename_with(~paste0("sd_", .), -PATIENT_ID)  

ts_rmssds <- dynamic_discrete %>% 
  group_by(PATIENT_ID, Period, Parameter) %>% 
  mutate(Value = sqrt(mean(diff(Value)^2))) %>% 
  unique(.) %>% 
  Pivot.Wide(.) %>% 
  rename_with(~paste0("rmssd_", .), -PATIENT_ID)  

# ts_iqrs <- dynamic_discrete %>% 
#   group_by(PATIENT_ID, Period, Parameter) %>% 
#   mutate(Value = IQR(Value, na.rm=TRUE)) %>% 
#   unique(.) %>% 
#   Pivot.Wide(.) %>% 
#   rename_with(~paste0("iqr_", .), -PATIENT_ID)
# 
# ts_maxs <- dynamic_discrete %>% 
#   group_by(PATIENT_ID, Period, Parameter) %>% 
#   mutate(
#     maximum = max(Value, na.rm = TRUE), 
#     Value = ifelse(maximum == -Inf, NA, maximum)
#   ) %>% 
#   unique(.) %>% 
#   Pivot.Wide(.) %>% 
#   rename_with(~paste0("max_", .), -PATIENT_ID)  
# 
# ts_mins <- dynamic_discrete %>% 
#   group_by(PATIENT_ID, Period, Parameter) %>% 
#   mutate(
#     minimum = min(Value, na.rm = TRUE), 
#     Value = ifelse(minimum == Inf, NA, minimum)
#   ) %>% 
#   unique(.) %>% 
#   Pivot.Wide(.) %>% 
#   rename_with(~paste0("min_", .), -PATIENT_ID)  
# 
# ts_ranges <- dynamic_discrete %>% 
#   group_by(PATIENT_ID, Period, Parameter) %>% 
#   mutate(
#     maximum = max(Value, na.rm = TRUE), 
#     maximum = ifelse(maximum == -Inf, NA, maximum), 
#     minimum = min(Value, na.rm = TRUE), 
#     minimum = ifelse(minimum == Inf, NA, minimum),  
#     Value = maximum-minimum
#     ) %>% 
#   unique(.) %>% 
#   Pivot.Wide(.) %>% 
#   rename_with(~paste0("range_", .), -PATIENT_ID)  
# 
# 
# ts_entropy <- dynamic_discrete %>% 
#   group_by(PATIENT_ID, Period, Parameter) %>% 
#   mutate(Value = entropy(na.omit(Value))) %>% 
#   unique(.) %>% 
#   Pivot.Wide(.) %>% 
#   rename_with(~paste0("entropy_", .), -PATIENT_ID)  



##### 03. Merge the wide features together ##### 
merged <- static %>% 
  left_join(ts_means, by = 'PATIENT_ID') %>% 
  left_join(ts_medians, by = 'PATIENT_ID') %>% 
  left_join(ts_sds, by = 'PATIENT_ID') %>% 
  left_join(ts_rmssds, by = 'PATIENT_ID') #%>% 
  # left_join(ts_iqrs, by = 'PATIENT_ID') %>% 
  # left_join(ts_maxs, by = 'PATIENT_ID') %>% 
  # left_join(ts_mins, by = 'PATIENT_ID') %>% 
  # left_join(ts_ranges, by = 'PATIENT_ID') %>% 
  # left_join(ts_entropy, by = 'PATIENT_ID') 

dim(merged)



##### 04. Conduct train/test splits #####
# get indices
train_idx <- c(caret::createDataPartition(labels$dep, p = 0.8, times=1, list = FALSE))

# extract train/test patient ids
train_patients <- labels[train_idx, 'RecordID']
test_patients <- labels[-train_idx, 'RecordID']

# check imbalances are about the same
labels %>% filter(RecordID %in% train_patients) %>% with(., table(dep)/nrow(.))
labels %>% filter(!RecordID %in% train_patients) %>% with(., table(dep)/nrow(.))

# partition the merged data
train_missing <- merged[merged$PATIENT_ID %in% train_patients, ] 
test_missing <- merged[!merged$PATIENT_ID %in% train_patients, ] 
dim(train_missing)
dim(test_missing)



##### 05. Impute the data with missForest #####
# extract ids and labels
train_ids_labs <- train_missing %>% left_join(labels, by = c('PATIENT_ID'='RecordID')) %>% select(PATIENT_ID, dep)
test_ids_labs <- test_missing %>% left_join(labels, by = c('PATIENT_ID'='RecordID')) %>% select(PATIENT_ID, dep)

# remove ids from dataframes
train_missing <- train_missing %>% select(-PATIENT_ID)
test_missing <- test_missing %>% select(-PATIENT_ID)

# partition the train and test sets by transformation
train_missing_means <- train_missing %>% select(contains(c('static', 'mean_')))
test_missing_means <- test_missing %>% select(contains(c('static', 'mean_')))

train_missing_medians <- train_missing %>% select(contains(c('static', 'med_')))
test_missing_medians <- test_missing %>% select(contains(c('static', 'med_')))

train_missing_sds <- train_missing %>% select(contains(c('static', 'sd_'))) %>% select(-contains('rmssd'))
test_missing_sds <- test_missing %>% select(contains(c('static', 'sd_'))) %>% select(-contains('rmssd'))

train_missing_rmssds <- train_missing %>% select(contains(c('static', 'rmssd_')))
test_missing_rmssds <- test_missing %>% select(contains(c('static', 'rmssd_')))

# train_missing_iqrs <- train_missing %>% select(contains(c('static', 'iqr_')))
# test_missing_iqrs <- test_missing %>% select(contains(c('static', 'iqr_')))
# 
# train_missing_maxs <- train_missing %>% select(contains(c('static', 'max_')))
# test_missing_maxs <- test_missing %>% select(contains(c('static', 'max_')))
# 
# train_missing_mins <- train_missing %>% select(contains(c('static', 'min_')))
# test_missing_mins <- test_missing %>% select(contains(c('static', 'min_')))
# 
# train_missing_ranges <- train_missing %>% select(contains(c('static', 'range_')))
# test_missing_ranges <- test_missing %>% select(contains(c('static', 'range_')))
# 
# train_missing_entropy <- train_missing %>% select(contains(c('static', 'entropy_')))
# test_missing_entropy <- test_missing %>% select(contains(c('static', 'entropy_')))

# fit the imputer objects to train data 
print('   Imputing means...')
imputer_means <- missForest(train_missing_means, maxiter = 3)

print('   Imputing medians...')
imputer_medians <- missForest(train_missing_medians, maxiter = 3)

print('   Imputing sds...')
imputer_sds <- missForest(train_missing_sds, maxiter = 3)

print('   Imputing rmssds...')
imputer_rmssds <- missForest(train_missing_rmssds, maxiter = 3)

# print('   Imputing IQRs...')
# imputer_iqrs <- missForest(train_missing_iqrs, maxiter = 3)
# 
# print('   Imputing maxs...')
# imputer_maxs <- missForest(train_missing_maxs, maxiter = 3)
# 
# print('   Imputing mins...')
# imputer_mins <- missForest(train_missing_mins, maxiter = 3)
# 
# print('   Imputing ranges...')
# imputer_ranges <- missForest(train_missing_ranges, maxiter = 3)
# 
# print('   Imputing entropy...')
# imputer_entropy <- missForest(train_missing_entropy, maxiter = 3)


# generate the imputed train sets
train_means <- imputer_means$ximp
train_medians <- imputer_medians$ximp
train_sds <- imputer_sds$ximp
train_rmssds <- imputer_rmssds$ximp
# train_iqrs <- imputer_iqrs$ximp
# train_maxs <- imputer_maxs$ximp
# train_mins <- imputer_mins$ximp
# train_ranges <- imputer_ranges$ximp
# train_entropy <- imputer_entropy$ximp

# generate the imputed test sets
test_means <- missForestPredict(imputer_means, test_missing_means)
test_medians <- missForestPredict(imputer_medians, test_missing_medians)
test_sds <- missForestPredict(imputer_sds, test_missing_sds)
test_rmssds <- missForestPredict(imputer_rmssds, test_missing_rmssds)
# test_iqrs <- missForestPredict(imputer_iqrs, test_missing_iqrs)
# test_maxs <- missForestPredict(imputer_maxs, test_missing_maxs)
# test_mins <- missForestPredict(imputer_mins, test_missing_mins)
# test_ranges <- missForestPredict(imputer_ranges, test_missing_ranges)
# test_entropy <- missForestPredict(imputer_entropy, test_missing_entropy)



##### 06. Save down the data #####
# ids and labels
write.csv(train_ids_labs, '../objects/train_ids_labs.csv', row.names = FALSE)
write.csv(test_ids_labs, '../objects/test_ids_labs.csv', row.names = FALSE)

# features 
write.csv(train_means, '../objects/train_means.csv', row.names = FALSE)
write.csv(test_means, '../objects/test_means.csv', row.names = FALSE)

write.csv(train_medians, '../objects/train_medians.csv', row.names = FALSE)
write.csv(test_medians, '../objects/test_medians.csv', row.names = FALSE)

write.csv(train_sds, '../objects/train_sds.csv', row.names = FALSE)
write.csv(test_sds, '../objects/test_sds.csv', row.names = FALSE)

write.csv(train_rmssds, '../objects/train_rmssds.csv', row.names = FALSE)
write.csv(test_rmssds, '../objects/test_rmssds.csv', row.names = FALSE)

# write.csv(train_iqrs, '../objects/train_iqrs.csv', row.names = FALSE)
# write.csv(test_iqrs, '../objects/test_iqrs.csv', row.names = FALSE)
# 
# write.csv(train_maxs, '../objects/train_maxs.csv', row.names = FALSE)
# write.csv(test_maxs, '../objects/test_maxs.csv', row.names = FALSE)
# 
# write.csv(train_mins, '../objects/train_mins.csv', row.names = FALSE)
# write.csv(test_mins, '../objects/test_mins.csv', row.names = FALSE)
# 
# write.csv(train_ranges, '../objects/train_ranges.csv', row.names = FALSE)
# write.csv(test_ranges, '../objects/test_ranges.csv', row.names = FALSE)
# 
# write.csv(train_entropy, '../objects/train_entropy.csv', row.names = FALSE)
# write.csv(test_entropy, '../objects/test_entropy.csv', row.names = FALSE)
