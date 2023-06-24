

sprintf("%0.1f%%", .7293827 * 100)


library(readxl)
library(dplyr)
library(tidyverse)


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

#************************************************************************************

df = df %>% group_by(Profile_ID) %>%
  mutate(order_month_year = format(as.Date(Date_of_Booking), "%Y-%m"),
         date_of_first_order = min(Date_of_Booking),
         Customer_status = case_when(Date_of_Booking == date_of_first_order ~"New",
                                                                         TRUE ~ "repeated"),
         no_of_days_frm_first_order = difftime(Date_of_Booking,date_of_first_order,units = "days"),
         repeat_order_type = case_when(no_of_days_frm_first_order == 0 ~"New_order",
                                       no_of_days_frm_first_order <=30 ~"repeat_order_within_30days",
                                       no_of_days_frm_first_order <=90 ~"repeat_order_within_90days",
                                       
                                       TRUE ~ "repeat_order_after_90days")) %>%
  ungroup()
 
                                            
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

#--------------------------------------------------------------------------------------------------


dim(df)

print(df[1:25,c(2,7,8,9,10,11)])
table(df$repeat_order_type)


library(ggplot2)


#Q1 - New user aquired every month
#------------------------------------
Q1 = df %>% 
  group_by(order_month_year,Customer_status) %>%
  summarise(total_order = length(Source)) %>%
  filter(Customer_status == 'New') %>%
  arrange(order_month_year)

ggplot(Q1, aes(y=total_order, x=order_month_year)) + 
  geom_bar(position="dodge", stat="identity",color='darkblue')+
  geom_text(aes(label=total_order), vjust=1.6, color="white",
            position = position_dodge(0.9), size=3.5)+
  scale_fill_manual(values=c("#E69F00"))+
#  scale_fill_brewer(palette="Dark2")+
  theme_minimal()


#Q2 - 30 day return_rate_of_customer aquired in december 2017
#----------------------------------------------------------------

tt = filter(df, order_month_year == "2017-12" & Customer_status == "New" )
dim(tt)


Q2 = count(tt[tt$repeat_order_type=="repeat_order_within_30days",])

print(paste("30 day return_rate_of_customer aquired in december 2017 is :",Q2))



#Q3 - What is the 90-day repeat rate of users acquired in Jan,Feb,March 2018?
#-------------------------------------------------------------

xt = filter(df, order_month_year %in% c("2018-01","2018-02","2018-03") & Customer_status == "New" )

dim(xt)

table(xt$order_month_year)

ids = c(xt$Profile_ID)


xt2 = filter(df, Profile_ID %in% ids )

dim(xt2)

Q3 = xt2 %>% group_by(Profile_ID) %>%
  summarize(month = order_month_year,
            new_User_count = length(Customer_status == "New_order" ),
            repeat_users_count_90days =length(repeat_order_type == "repeat_order_within_90days"))%>%
  filter(month %in% c("2018-01","2018-02","2018-03"))

  
Q3








Q3 = df %>% filter(Profile_ID %in% ids) %>%
  group_by(order_month_year) %>%
  summarise(month = order_month_year,
            total_orders= length(repeat_order_type=="New"),
            percent_orders = (n()/length(df$Profile_ID))*100)%>%
  filter()


head(Q2)







Q3 = df %>% 
  group_by(order_month_year,repeat_order_type) %>%
  summarise(total_order = length(Source)) %>%
  filter(order_month_year %in% c('2018-01','2018-02','2018-03'),
         repeat_order_type %in% c("repeat_order_within_90days")) %>%
  arrange((order_month_year))

Q3

ggplot(Q3, aes(fill=repeat_order_type, y=total_order, x=order_month_year)) + 
  geom_bar(position="dodge", stat="identity")







str(df)



#---------------------------------------------------------------------































