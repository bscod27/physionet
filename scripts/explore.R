library(tidyverse)
library(table1)
source('utils.R')


##### 01. Load the data #####
# labels
labels <- read.table('../data/Outcomes-a.txt', header=TRUE, sep=',') %>% select(RecordID, dep=In.hospital_death)
str(labels)
table(complete.cases(labels))

# features
features <- read.csv('../data/seta_data.csv') %>% 
  # make some simple mutations - turn time into numeric and rename variables
  mutate(
    Value = ifelse(Value == -1, NA, Value), 
    HR = as.numeric(substr(Time, 1, 2)) + as.numeric(substr(Time, 4, 5))/60, # convert to hours, 
    Parameter = case_when(
      Parameter %in% c(
        'Age', 'Gender', 'Height', 'ICUType', 'RecordID'
        ) ~ paste0('static_', Parameter), 
      TRUE ~ paste0('dynamic_', Parameter)
      )
    ) %>% 
  # arrange by the groupings 
  arrange(PATIENT_ID, Parameter, Time) %>% 
  # combine duplicate measurements at the same time point by taking their average
  group_by(PATIENT_ID, Parameter, Time) %>% 
  mutate(Value = mean(Value))

head(features)
table(features$Parameter, is.na(features$Value))

# save down mutated features and labels for future use
save(features, labels, file = '../objects/features_labels.RData')



##### 02. Define data frames for EDA #####

# collapsed data (patient-level)
df_collapsed <- features %>% 
  # get the mean value of each variable for each patient
  group_by(PATIENT_ID, Parameter) %>% 
  summarize(mean=mean(Value, na.rm=TRUE)) %>% 
  # pivot to wide format
  pivot_wider(
    id_cols = PATIENT_ID, 
    names_from = Parameter, 
    values_from = mean
  ) %>% 
  # join the labeled data
  left_join(labels, by = c('PATIENT_ID'='RecordID')) %>% 
  # reorder columns
  select(PATIENT_ID, dep, contains(c('static', 'dynamic'))) %>% 
  as.data.frame(.)

head(df_collapsed)

# long format
df_continuous <- features %>% 
  # deal with dups
  group_by(PATIENT_ID, HR, Parameter) %>% 
  summarize(Value = mean(Value)) %>% 
  # pivot wider
  pivot_wider(
    id_cols = c(PATIENT_ID, HR),
    names_from = Parameter, 
    values_from = Value
  ) %>% 
  select(PATIENT_ID, HR, contains(c('static', 'dynamic')))

head(df_continuous)



##### 03. Explore basic descriptives #####
# dimensions
dim(df_collapsed)

# basic descriptives 
summary(df_collapsed %>% select(-PATIENT_ID))

# outcome imbalance
table(df_collapsed$dep)/nrow(df_collapsed) # proportions 
table(df_collapsed$dep) # counts 

# missingness
sort(Compute.Missingness(df_collapsed)) # proportions
sort(Compute.Missingness(df_collapsed)) * nrow(df_collapsed) # counts 



##### 04. Describe baseline patient features, stratified by outcome #####
patient_chars <- 
  table1(
  ~ static_Age + static_Gender + static_Height + static_ICUType | 
    factor(dep, levels=0:1, labels=c('Survived', 'Died')), 
  
  data = df_collapsed %>% 
    select(PATIENT_ID, dep, contains('static_')) %>% 
    mutate(
      static_Age = case_when(
        static_Age < 50 ~ '<50', 
        static_Age >= 50 & static_Age < 60~ '50-59', 
        static_Age >= 60 & static_Age < 69~ '60-69', 
        static_Age >= 70 & static_Age < 79~ '70-79', 
        TRUE ~ '80+'
      ), 
      static_Gender = case_when(
        static_Gender == 0 ~ 'Female', 
        static_Gender == 1 ~ 'Male', 
        TRUE ~ NA
      ), 
      static_ICUType = case_when(
        static_ICUType == 1 ~ 'Coronary Care Unit', 
        static_ICUType == 2 ~ 'Cardiac Surgery Recovery Unit', 
        static_ICUType == 3 ~ 'Medical ICU', 
        TRUE ~ 'Surgical ICU'
      ), 
      static_Height = case_when(
        is.na(static_Height) ~ 'Missing', 
        static_Height < 165 ~ 'Short (<165 cm)', 
        static_Height >= 165 & static_Height < 180 ~ 'Average (165-180 cm)', 
        TRUE ~ 'Tall (185+ cm)'
        ) %>% 
        factor(., levels = c('Short (<165 cm)', 'Average (165-180 cm)', 'Tall (185+ cm)', 'Missing'))
      ),
  overall = FALSE, 
  extra.col=list(`P-value`=P.Value),
) 
write.csv(as.data.frame(patient_chars), '../results/patient_chars.csv', row.names = FALSE)

  

##### 05. Examine the distribution of dynamic features ##### 
png('../results/boxplots_distributions.png', units = 'in', width = 12, height = 7, res = 300)
boxplot(
  df_collapsed %>% 
    as.data.frame(.) %>% 
    select(contains('dynamic')),
  ylab = 'Value',
  ylim=c(-10,100), 
  las = 2, 
  cex.axis = 0.5
)
dev.off()



##### 06. Perform outlier detection ##### 
png('../results/boxplots_outliers.png', units = 'in', width = 12, height = 7, res = 300)
boxplot(
  df_collapsed %>% 
    as.data.frame(.) %>% 
    select(contains('dynamic')) %>%
    scale(.), 
  ylim=c(-10,10), 
  ylab = 'Standardized Value',
  las = 2, 
  cex.axis = 0.5
  )
abline(h=3, lty=2, col='red')
abline(h=-3, lty=2, col='red')
dev.off()


##### 07. Understand missingness patterns #####
png('../results/heatmap_missingness.png', units = 'in', width = 5, height = 5, res = 300)
heatmap(
  x = df_collapsed %>% 
    mutate_all(function(i) ifelse(is.na(i), 1, 0)) %>% 
    as.data.frame() %>% 
    select(-PATIENT_ID, -dep) %>% 
    as.matrix(),
  Rowv = NA,                 
  Colv = NA, 
  labRow = NA,
  col = c('white', 'black'),
  cexCol = 0.55 
)
dev.off()



##### 08. Examine correlations between dynamic variables #####
cor_matrix <- cor(
  df_collapsed %>% 
    as.data.frame() %>% 
    select(dep, contains('dynamic')) %>% 
    select(-'dynamic_MechVent'), 
  use='pairwise.complete.obs'
)
write.csv(cor_matrix, '../results/correlation_matrix.csv')



##### 09. Corroborate modeling strategy by looking at longitudinal missingness #####
png('../results/heatmap_missingness_long.png', units = 'in', width = 5, height = 5, res = 300)
heatmap(
  x=df_continuous %>% 
    mutate_all(function(i) ifelse(is.na(i), 1, 0)) %>% 
    arrange(PATIENT_ID, HR) %>% 
    as.data.frame(.) %>% 
    select(-PATIENT_ID, -HR) %>% 
    as.matrix(), 
  Rowv = NA,                 
  Colv = NA, 
  labRow = NA,
  col = c('white', 'black'),
  cexCol = 0.55 # Reduce column label size
) 
dev.off()



##### 10. Looking for cycles to define discret time windows #####
# scale the features
dynamic_scaled <- features %>% 
  filter(str_detect(Parameter, 'dynamic')) %>% 
  group_by(Parameter) %>% 
  mutate(
    scaled = scale(Value),
    scaled = ifelse(scaled > 3, 3,ifelse(scaled < -3, -3, scaled)) # clipped for visualization purposes
    )
dynamic_scaled$Parameter <- gsub('_', '', unlist(str_match_all(dynamic_scaled$Parameter, '_.*')))

png('../results/spaghetti_cycles.png', units = 'in', width = 10, height = 7, res=150)
dynamic_scaled %>% 
  filter(Parameter != 'MechVent') %>% 
  left_join(labels, by = c('PATIENT_ID' = 'RecordID')) %>% 
  ggplot(aes(
    x = HR, 
    y = scaled, 
    color = factor(dep, levels = 0:1, labels = c('Survived', 'Died'))
  )) + 
  geom_smooth(method = 'gam', formula = y ~ s(x), se = FALSE, size = 1) +
  facet_wrap(Parameter ~ ., nrow = 4) +
  theme(
    strip.text = element_text(size = 8),
    axis.text.x = element_text(size = 6),  
    legend.position = 'top',               
    legend.direction = 'horizontal',       
    legend.title = element_text(size = 8), 
    legend.text = element_text(size = 6)   
  ) + 
  scale_color_discrete(name = 'Outcome') +
  xlab('Hours from Baseline') +
  ylab('Standardized Value') +
  coord_cartesian(ylim = c(-0.5, 1.5))
dev.off()



##### 11. Plot the cardinality consequences of discretizing into bins #####
n <- nrow(df_collapsed)*0.8 # training data will have 80% of observations
no_static <- df_collapsed %>% select(contains('static')) %>% ncol + 1 # +1 for one-hotting ICU Type and getting rid of recordID
no_dynamic <- df_collapsed %>% select(contains('dynamic')) %>% ncol 
no_periods <- 1:8 # number of discretization periods
no_trans <- 1:4 # number of time series transformations
stats <- data.frame()
for (i in no_trans) {
  temp <- data.frame(
    transformations=i, 
    periods=no_periods,
    features = no_static + no_dynamic*no_periods*i
  )
  stats <- rbind(stats, temp)
}

png('../results/lineplot_cardinality.png', units = 'in', width = 5, height = 5, res = 300)
stats %>% 
  ggplot(aes(x=periods, y=log(features), group=transformations)) + 
  geom_smooth(aes(color=factor(transformations)), se=FALSE) + 
  theme_classic() + 
  geom_hline(yintercept=log(n/10), lty=2) +
  geom_vline(xintercept=2, lty=2, col='grey') + 
  geom_vline(xintercept=4, lty=2, col='grey') + 
  geom_vline(xintercept=6, lty=2, col='grey') + 
  geom_vline(xintercept=8, lty=2, col='grey') + 
  annotate('text', x = max(no_periods)-0.5, y = log(n/10), 
           label = 'log(n/10)', vjust = -0.5, hjust = 1) +
  theme(
    legend.position = 'top',               
    legend.direction = 'horizontal'
  ) + 
  scale_color_discrete(name = 'Number of Time Series Features') +  
  xlab('Number of Discrete Time Bins') +
  ylab('log(Number of Features)') 
dev.off()



##### 12. Calculate the data availability consequences of discretizing into bins #####
# form bins
bins_1 <- df_collapsed %>% 
  mutate(bin=1) %>% 
  as.data.frame(.) %>% 
  select(-contains('static')) 

bins_2 <- df_continuous %>% 
  mutate(bin = ifelse(HR <= 24, 1, 2)) %>% 
  as.data.frame(.) %>% 
  select(-HR, -contains('static'))

bins_4 <- df_continuous %>% 
  mutate(
    bin = ifelse(HR <= 12, 1, 
          ifelse(HR > 12 & HR <= 24, 2, 
          ifelse(HR > 24 & HR <= 36, 3, 4)))
    ) %>% 
  as.data.frame(.) %>% 
  select(-HR, -contains('static'))

bins_6 <- df_continuous %>% 
  mutate(
    bin = ifelse(HR <= 8, 1, 
          ifelse(HR > 8 & HR <= 16, 2, 
          ifelse(HR > 16 & HR <= 24, 3, 
          ifelse(HR > 24 & HR <= 32, 4, 
          ifelse(HR > 32 & HR <= 40, 5, 6)))))
    ) %>% 
  as.data.frame(.) %>% 
  select(-HR, -contains('static'))

bins_8 <- df_continuous %>% 
  mutate(
    bin = ifelse(HR <= 6, 1, 
          ifelse(HR > 6 & HR <= 12, 2, 
          ifelse(HR > 12 & HR <= 18, 3, 
          ifelse(HR > 18 & HR <= 24, 4, 
          ifelse(HR > 24 & HR <= 30, 5, 
          ifelse(HR > 30 & HR <= 36, 6, 
          ifelse(HR > 36 & HR <= 42, 7, 8)))))))
    ) %>% 
  as.data.frame(.) %>% 
  select(-HR, -contains('static'))

# count number bins with at least one observation
Count.Completeness <- function(df) {
  temp <- df %>% 
    as.data.frame(.) %>% 
    select(PATIENT_ID, bin, everything()) %>% 
    group_by(PATIENT_ID, bin) %>% 
    mutate_all(., function(i) ifelse(sum(!is.na(i))>=1, 1, 0)) %>% 
    unique(.)
  
  stats <- temp %>% 
    group_by(PATIENT_ID, bin) %>% 
    summarize(completeness=rowSums(across(contains('dynamic')))/(ncol(temp)-2)) %>% 
    group_by(bin) %>% 
    summarize(completeness=mean(completeness))
  
  return(stats)
}
  
out <- Count.Completeness(bins_8) %>% 
  left_join(Count.Completeness(bins_6), by = 'bin') %>% 
  left_join(Count.Completeness(bins_4), by = 'bin') %>% 
  left_join(Count.Completeness(bins_2), by = 'bin') %>% 
  left_join(Count.Completeness(bins_1), by = 'bin')

cols <- paste0('Bin ', as.integer(out[,1]$bin))
reordered <- cbind(cols, cbind(out[,6], out[,5], out[,4], out[,3], out[,2]))

final <- 
  rbind(
    reordered,
    cbind(
      cols='Average',
      reordered %>% 
        mutate_if(is.numeric, function(i) mean(i,na.rm=TRUE)) %>% 
        select(-cols) %>% 
        unique(.)
    )
  )

colnames(final) <- c('cols', 'discrete_1', 'discrete_2', 'discrete_4', 'discrete_6', 'discrete_8')
final

write.csv(final,'../results/completeness_perbin.csv', row.names = FALSE)
