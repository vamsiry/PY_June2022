

library(readxl)
library(dplyr)
library(scales)
library(tidyverse)
library(stringr) 
library(ggplot2) 
library(ggthemes) 


df = read_excel("C://Users//vamsi//Desktop//PEP//Pep Analytics Test.xlsx", sheet = "Data")

dim(df)
glimpse(df)

names(df)[names(df) == "Profile ID"] <- "Profile_ID"
names(df)[names(df) == "Slot of Booking (Hour of the Day)"] <- "Slot_of_Booking_HOD"

class(df)
str(df)

df$Profile_ID = as.factor(df$Profile_ID)
df$Source = as.factor(df$Source)
df$Slot_of_Booking_HOD = as.factor(df$Slot_of_Booking_HOD)

str(df)

#------------------------------------------------------------------------------------------
#Data transformation
#----------------------------
df = df %>% group_by(Profile_ID) %>%
  mutate(user_join_month_year = min(format(as.Date(Date_of_Booking), "%Y-%m")),
         user_repeat_order_month_year = format(as.Date(Date_of_Booking), "%Y-%m")) %>%
  ungroup()





# create reference data frame of total users for each cohort group
base_cohort_df = df %>% group_by(user_join_month_year) %>%
  summarise(New_users_aquired = n_distinct(Profile_ID))

base_cohort_df %>% head(20)





# create purchase activity data frame
activity_cohort_df = df %>% group_by(user_join_month_year, user_repeat_order_month_year) %>%
  summarise(repeate_BuyingUsers = n_distinct(Profile_ID))

activity_cohort_df %>% head(20)





# join activity_cohort_df and base_cohort_df
user_cohort_df = inner_join(activity_cohort_df, base_cohort_df, 
                            by = 'user_join_month_year')
user_cohort_df %>% head(20)




# create  OrderMonth as integer for ploting on x-axis
user_cohort_df = user_cohort_df %>% 
  group_by(user_join_month_year) %>% 
  mutate(MonthNumber = 1:n())

user_cohort_df %>% head(20)



#----------------------------------------------------------------
#Q1 - New user aquired every month
#------------------------------------
Q1 = user_cohort_df[user_cohort_df$MonthNumber==1,c(1,4)]

Q1

ggplot(data=Q1, aes(x=user_join_month_year, y=New_users_aquired)) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=New_users_aquired), vjust=1.6, color="white", size=3.5)+
  theme_minimal()



#----------------------------------------------------------------
#Q2 - 30 day return_rate_of_customer aquired in december 2017
#----------------------------------------------------------------
Q2 = filter(user_cohort_df,user_join_month_year == "2017-12" & MonthNumber == "2" )


Q2_30 = label_percent()(Q2$repeate_BuyingUsers/Q2$New_users_aquired)


cat(paste("30 day return_rate_of_customer aquired in december 2017 is :",Q2_30))




#----------------------------------------------------------------
#Q3 - What is the 90-day repeat rate of users acquired in Jan,Feb,March 2018?
#-------------------------------------------------------------
d3 = filter(user_cohort_df, 
            user_join_month_year %in% c("2018-01","2018-02","2018-03") & 
              MonthNumber %in% c("2","3","4" ))
d3

Q3 = d3 %>% group_by(user_join_month_year) %>%
  summarize(New_users_aquired = mean(New_users_aquired),
            Repeat_count_90days = sum(repeate_BuyingUsers),
            Repeat_Rate_90days = 
              label_percent()(sum(repeate_BuyingUsers)/mean(New_users_aquired)))

Q3



Q3_2 = Q3[,c(1,2,3)]
Q3_22 = melt(Q3_2,id.vars = 'user_join_month_year')


ggplot(Q3_22, aes(x=user_join_month_year, y=value, fill=variable)) +
  geom_bar(stat='identity', position='dodge')+
  geom_text(aes(label=value),hjust=0.5, vjust=3, size=3.5,color="white",
            position = position_dodge(width = 1))+
  theme_minimal()



#-------------------------------------------------------------
# Q4 -  model to predict the 90-day repeat of users 
#---------------------------------------------------

#Built time series-Holt winter linear model to predict 
#90 day retention users

#Data transformed monthly to weekly as we have only 
#12 samples of monthly data after transformation


#data transformed to weekly new users and calculated their
#90 day retention rate as target variable to predict/forecast
#the last 11 weeks of data as the last 11 weeks data do not
#have following data to calculate and include in the 
#training data


#Data preparation/Transformation
TS_data = df %>% group_by(Profile_ID) %>%
  mutate(New_user_join_wk_yr = min(format(as.Date(Date_of_Booking), "%Y-%U")),
         user_repeat_order_wk_yr = format(as.Date(Date_of_Booking), "%Y-%U")) %>%
  ungroup()

dim(TS_data)

TS_data[,c(7,8,9,10)] %>% head(20)




# create reference data frame of total users for each group
base_ts_data = TS_data %>% 
  group_by(New_user_join_wk_yr) %>%
  summarise(New_users_aquired = n_distinct(Profile_ID))

base_ts_data %>% head(20)




# create purchase activity data frame
activity_ts_data = TS_data %>% 
  group_by(New_user_join_wk_yr, user_repeat_order_wk_yr) %>%
  summarise(repeate_Buying_Users_count = n_distinct(Profile_ID))

activity_ts_data %>% head(20)




# join activity_cohort_df and base_cohort_df
user_ts_data = inner_join(activity_ts_data, base_ts_data, 
                            by = "New_user_join_wk_yr")

user_ts_data %>% head(20)




# transform Orderweek to integer
user_ts_data_df = user_ts_data %>% 
  group_by(New_user_join_wk_yr) %>% 
  mutate(weeknumber = 1:n())

user_ts_data_df %>% head(20)



ts_model_data = user_ts_data_df %>% 
  filter(weeknumber %in% c(2:10 )) %>%
  group_by(New_user_join_wk_yr) %>%
  summarize(New_users_aquired = mean(New_users_aquired),
            Repeat_90day_user_count = sum(repeate_Buying_Users_count),
            Repeat_90day_user_Percent = 
              label_percent()(sum(repeate_Buying_Users_count)/mean(New_users_aquired)))

dim(ts_model_data)


p1 = ts_model_data[,c(1,2,3)]
head(p1)

plot_data = melt(p1,id.vars = 'New_user_join_wk_yr')
plot_data %>% head(10)

ggplot(plot_data, aes(x=New_user_join_wk_yr, y=value, fill=variable)) +
  geom_bar(stat='identity', position='dodge')+
  geom_text(aes(label=value),hjust=0.5, vjust=1, size=2.5,color="white",
            position = position_dodge(width = 0.5))+
  theme_minimal()



#modeling
#----------
#Time series model of 90day reapet user forcast of new user aquired weekly

library(forecast)

train_data = ts_model_data[1:44,c(1,3)]

train_data %>% head(20)

#Creating TS Object
ts_object <- ts(train_data$Repeat_90day_user_count, start=c(2017, 48), end=c(2018, 39), frequency=52)

train <- window(ts_object, start=c(2017, 48), end=c(2018, 29))

test <- window(ts_object, start=c(2018, 30))


#Holt winter Models
#------------------
# identify optimal beta parameter
beta <- seq(.0001, .5, by = .001)

RMSE <- NA

for(i in seq_along(beta)) {
  fit <- holt(train, beta = beta[i], h = 11)
  RMSE[i] <- accuracy(fit, test)[2,2]
}


# convert to a data frame and idenitify min alpha value
beta.fit <- data_frame(beta, RMSE)
beta.min <- filter(beta.fit, RMSE == min(RMSE))

beta.min

# plot RMSE vs. beta
ggplot(beta.fit, aes(beta, RMSE)) +
  geom_line() +
  geom_point(data = beta.min, aes(beta, RMSE), size = 2, color = "blue")  



#Fitting model with optimal beta
fit <- holt(train, beta = 0.282, h = 11)

autoplot(fit)

autoplot(fit) + autolayer(fitted(fit))


fit$model

accuracy(fit, test)




#------------------------------------------------------------------------
#Q5 Plot the distribution of users by frequency of their 90-day repeat 
#(Number of times user repeated within first 90 days)
#-------------------------------------------------------

Q5 = user_cohort_df %>% 
  filter(MonthNumber %in% c("2","3","4" )) %>%
  group_by(user_join_month_year) %>%
  summarize(New_users_aquired = mean(New_users_aquired),
            Repeat_count_90days = sum(repeate_BuyingUsers),
            Repeat_Rate_90days = 
              label_percent()(sum(repeate_BuyingUsers)/mean(New_users_aquired)))

Q5 %>% head(20)

Q5_2 = Q5[,c(1,2,3)]
Q5_22 = melt(Q5_2,id.vars = 'user_join_month_year')


ggplot(Q5_22, aes(x=user_join_month_year, y=value, fill=variable)) +
  geom_bar(stat='identity', position='dodge')+
  geom_text(aes(label=value),hjust=0.5, vjust=1, size=3.5,color="white",
            position = position_dodge(width = 1))+
  theme_minimal()



#----------------------------------------------------------------------------
#Q6 :bounus Cohort analysis
#-------------------------------

user_cohort_df %>% head(10)

# create base dataframe for heat map visualization
cohort_heatmap_df = user_cohort_df %>% 
  select(user_join_month_year, MonthNumber, repeate_BuyingUsers) %>%
  spread(MonthNumber, repeate_BuyingUsers)

cohort_heatmap_df %>% head(20)



# the percentage version of the dataframe
cohort_heatmap_df_pct = data.frame(
  cohort_heatmap_df$user_join_month_year,
  cohort_heatmap_df[,2:ncol(cohort_heatmap_df)] / cohort_heatmap_df[["1"]]
)

cohort_heatmap_df_pct %>% head(20)

# assign the same column names
colnames(cohort_heatmap_df_pct) = colnames(cohort_heatmap_df)




# melt the dataframes for plotting
plot_data_abs = gather(cohort_heatmap_df, "MonthNumber",
                       "BuyingUsers", 2:ncol(cohort_heatmap_df))
plot_data_pct = gather(cohort_heatmap_df_pct, "MonthNumber", 
                       "Retention", 2:ncol(cohort_heatmap_df_pct))

plot_data_abs %>% head(20)
plot_data_pct %>% head(20)



# prepare label names containing absolute number of buyers for the
#first month and retention percentages for the rest months
label_names = c(plot_data_abs$BuyingUsers[1:(ncol(cohort_heatmap_df)-1)],
                plot_data_pct$Retention[(ncol(cohort_heatmap_df_pct)):(nrow(plot_data_pct))])



# adding percentage labels
print_with_percent <- function(n) {
  case_when( n <= 1  ~ sprintf("%1.0f %%", n*100),
             n >  1  ~ as.character(n),
             TRUE    ~ " ") 
}


# create dataframe ready for plotting
plot_data = data.frame(
  CohortGroup = plot_data_pct$user_join_month_year,
  MonthNumber = plot_data_pct$MonthNumber,
  Retention = plot_data_pct$Retention,
  Label = print_with_percent(label_names)
)


plot_data = plot_data %>% 
  group_by(CohortGroup) %>% 
  mutate(MonthNumber_2 = 1:n())

plot_data$MonthNumber = as.numeric(plot_data$MonthNumber)



#Cohort Chat
#----------------
ggplot(plot_data, 
       aes(x = MonthNumber_2, 
           y = reorder(CohortGroup, desc(CohortGroup)))) +
  geom_raster(aes(fill = Retention)) +
  scale_fill_continuous(guide = FALSE,type = "gradient",
                        low = "orange", high = "darkblue") +
  scale_x_continuous(breaks = seq(from = 2, to = 15, by = 1),
                     expand = c(0,0)) +
  geom_text(aes(label = Label), color = "white") +
  xlab("Cohort age") + ylab("Cohort Group - Customer join month") + 
  ggtitle(paste("Retention table (Cohort) monthly repeat rate of users acquired from aquisition month"))












