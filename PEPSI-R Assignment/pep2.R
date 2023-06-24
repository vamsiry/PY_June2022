


library(readxl)
library(dplyr)
library(tidyverse)

# import libraries
library(dplyr) #handy data manipulation
library(ggplot2) #our today's super star
library(stringr) #to manipulate string date
library(ggthemes) #many nice themes
library(mdthemes) #handy text in plot formatting
library(gghighlight) #will abuse it a bit to show nice label


install.packages("gghighlight")

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
df = df %>% group_by(Profile_ID) %>%
  mutate(user_join_month_year = min(format(as.Date(Date_of_Booking), "%Y-%m")), 
         user_order_month_year = format(as.Date(Date_of_Booking), "%Y-%m"),
         date_of_first_order = min(Date_of_Booking),
         Customer_status_in_month_year = case_when(Date_of_Booking == date_of_first_order ~"New",
                                                   TRUE ~ "repeated"),
         no_of_days_frm_first_order = difftime(Date_of_Booking,date_of_first_order,units = "days"),
         repeat_order_type = case_when(no_of_days_frm_first_order == 0 ~"New_order",
                                       no_of_days_frm_first_order <=30 ~"repeat_order_within_30days",
                                       no_of_days_frm_first_order <=90 ~"repeat_order_within_90days",
                                       
                                       TRUE ~ "repeat_order_after_90days")) %>%
  ungroup()


dim(df)

print(df[1:25,c(2,7,8,9,10,11,12)])
table(df$repeat_order_type)


#--------------------------------------------------------------
# create reference data frame of total users for each cohort group
base_cohort_df = df %>% group_by(user_join_month_year) %>%
  summarise(
    TotalUsers = n_distinct(Profile_ID)
  )

base_cohort_df %>% head(20)



# create purchase activity data frame
activity_cohort_df = df %>% group_by(user_join_month_year, user_order_month_year) %>%
  summarise(BuyingUsers = n_distinct(Profile_ID))

activity_cohort_df %>% head(20)



# join activity_cohort_df and base_cohort_df
user_cohort_df = inner_join(activity_cohort_df, base_cohort_df, 
                            by = 'user_join_month_year')



user_cohort_df %>% head(20)

#-------------------------------------------------------------
#Notice OrderMonth column is still in string format as above. 
#For plotting, we want it to be in integer format to become the x-axis.

# transform OrderMonth to integer
user_cohort_df = user_cohort_df %>% 
  group_by(user_join_month_year) %>% 
  mutate(MonthNumber = 1:n())

user_cohort_df %>% head(20)


#----------------------------------------------------------------
# create base dataframe for heat map visualization
cohort_heatmap_df = user_cohort_df %>% 
  select(user_join_month_year, MonthNumber, BuyingUsers) %>%
  spread(MonthNumber, BuyingUsers)


# inspect data
cohort_heatmap_df %>% head(5)


#-------------------------------------------------------------
# the percentage version of the dataframe
cohort_heatmap_df_pct = data.frame(cohort_heatmap_df$user_join_month_year,
  round(cohort_heatmap_df[,2:ncol(cohort_heatmap_df)] / 
    cohort_heatmap_df[["1"]]*100,2))

# assign the same column names
colnames(cohort_heatmap_df_pct) = colnames(cohort_heatmap_df)

# inspect data
cohort_heatmap_df_pct %>% head(15)

#------------------------------------------------------------
# melt the dataframes for plotting
plot_data_abs = gather(cohort_heatmap_df, "MonthNumber", "BuyingUsers", 2:ncol(cohort_heatmap_df))
plot_data_pct = gather(cohort_heatmap_df_pct, "MonthNumber", "Retention", 2:ncol(cohort_heatmap_df_pct))

plot_data_abs %>% head(20)

#-------------------------------------------------------------
# prepare label names containing absolute number of buyers for the first 
#month and retention percentages for the rest months
label_names = c(plot_data_abs$BuyingUsers[1:(ncol(cohort_heatmap_df)-1)],
                plot_data_pct$Retention[(ncol(cohort_heatmap_df_pct)):(nrow(plot_data_pct))])


#-------------------------------------------------------------------------
#Finally, we put everything together into a dataframe ready for plotting.
# beautify percentage labels
beauty_print <- function(n) {
  case_when( n <= 1  ~ sprintf("%1.0f %%", n*100),
             n >  1  ~ as.character(n),
             TRUE    ~ " ") # for NA values, skip the label
}

# create dataframe ready for plotting
plot_data = data.frame(
  user_join_month_year = plot_data_pct$user_join_month_year,
  MonthNumber = plot_data_pct$MonthNumber,
  Retention = plot_data_pct$Retention,
  Label = beauty_print(label_names)
)

plot_data$MonthNumber = as.numeric(plot_data$MonthNumber)

plot_data %>% head(25)


#----------------------------------------------------------------------
# plotting heatmap
ggplot(plot_data) +
  geom_raster(aes(x = MonthNumber,
                  y = reorder(user_join_month_year, desc(user_join_month_year)),
                  fill = Retention)) +
  scale_fill_continuous(guide = FALSE, type = "gradient",
                        low = "deepskyblue", high = "darkblue") +
  scale_x_continuous(breaks = seq(from = 1, to = 15, by = 1),
                     expand = c(0,0)) +
  geom_text(aes(x = MonthNumber,
                y = reorder(user_join_month_year, desc(user_join_month_year)),
                label = Label), col = "white")


#-----------------------------------------------------------------------












