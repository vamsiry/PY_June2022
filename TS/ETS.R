


#Step 1: Load R Packages 
#-------------------------

library(data.table)
library(dplyr)
library(ggplot2)
library(lubridate)
library(ggplot2)
library(forecast)
library(tseries)


setwd("C:/Users/rvamsikrishna/Desktop/TS/Pay_Play_TS_Data")


# read data
df = fread("Pay_Chat_PST.csv")

names(df) = c("Date", "Hour_of_Day", "Contact_Volume")

df$Date = as.Date(df$Date,"%m/%d/%Y")
df$Hour_of_Day = as.numeric((df$Hour_of_Day))

str(df)
summary(df)
table(is.na(df))


#as.Date("2015-08-01") - as.Date("2018-07-31")
#1096*24

t = seq(as.POSIXct("2015-08-01 00:00:00"), as.POSIXct("2018-07-31 23:00:00"), by="hour")
dt  = data.frame(date_time = (t), Date = date(t), Hour_of_Day = hour(t), zero_vloume = 0)


dim(dt)
dim(df)


new_df = merge(dt, df, by = c("Date","Hour_of_Day"), all.x = TRUE )
table(is.na(new_df$Contact_Volume))
new_df$Contact_Volume[is.na(new_df$Contact_Volume)] = 0
new_df$zero_vloume = NULL
summary(new_df)


new_df_1 = new_df

head(new_df)
new_df = new_df[,c(3,1,2,4)]
head(new_df)


#---------------------------------------------------------------------------------------------------------------
aggr_new_df = data.table(new_df)

head(aggr_new_df)
# Aggregate  data day-wise
aggr_new_df = aggr_new_df[,list(Count = sum(Contact_Volume)), by = Date]
dim(aggr_new_df)
head(aggr_new_df)
class(aggr_new_df)
table(is.na(aggr_new_df))
table(is.na(new_df))


# Examine Your Data
#-------------------------
head(aggr_new_df)
library(lubridate)

aggr_new_df$day_of_week = wday(aggr_new_df$Date) # Sun=1, Sat=7
aggr_new_df$day_of_mnth = day(aggr_new_df$Date)
aggr_new_df$week_num_of_mnth = ceiling(day(aggr_new_df$Date) / 7)
aggr_new_df$week_num_of_year = week(aggr_new_df$Date)
aggr_new_df$month_num = month(aggr_new_df$Date)

head(aggr_new_df)

plot(aggr_new_df$Count ~ aggr_new_df$day_of_week, data = aggr_new_df)
plot(aggr_new_df$Count ~ aggr_new_df$day_of_mnth, data = aggr_new_df)
plot(aggr_new_df$Count ~ aggr_new_df$week_num_of_mnth, data = aggr_new_df)
plot(aggr_new_df$Count ~ aggr_new_df$week_num_of_year, data = aggr_new_df)
plot(aggr_new_df$Count ~ aggr_new_df$month_num, data = aggr_new_df)
plot(aggr_new_df$Count ~ aggr_new_df$Date, data = aggr_new_df)

ggplot(aggr_new_df, aes(Date, Count)) + geom_line() + scale_x_date('month')  + ylab("Daily pay chat volume") +
  xlab("")


new_df$day_of_week = wday(new_df$Date) # Sun=1, Sat=7
new_df$day_of_mnth = day(new_df$Date)
new_df$week_num_of_mnth = ceiling(day(new_df$Date) / 7)
new_df$week_num_of_year = week(new_df$Date)
new_df$month_num = month(new_df$Date)

head(new_df)

plot(new_df$Contact_Volume ~ new_df$Hour_of_Day, data = new_df)
plot(new_df$Contact_Volume ~ new_df$day_of_week, data = new_df)
plot(new_df$Contact_Volume ~ new_df$day_of_mnth, data = new_df)
plot(new_df$Contact_Volume ~ new_df$week_num_of_mnth, data = new_df)
plot(new_df$Contact_Volume ~ new_df$week_num_of_year, data = new_df)
plot(new_df$Contact_Volume ~ new_df$month_num, data = new_df)
plot(new_df$Contact_Volume ~ new_df$Date, data = new_df)

ggplot(new_df, aes(Date, Contact_Volume)) + geom_line() + scale_x_date('Hour_of_Day')  + ylab("Daily pay chat volume") + xlab("")


ggplot(new_df, aes(Hour_of_Day, Contact_Volume)) + geom_line()  + ylab("Daily pay chat volume") + xlab("") +
  scale_x_continuous("Hour_of_Day", labels = as.character(new_df$Hour_of_Day), breaks = new_df$Hour_of_Day)

ggplot(new_df, aes(day_of_week, Contact_Volume)) + geom_line()  + ylab("Daily pay chat volume") + xlab("")

ggplot(new_df, aes(day_of_mnth, Contact_Volume)) + geom_line()  + ylab("Daily pay chat volume") + xlab("")

ggplot(new_df, aes(week_num_of_mnth, Contact_Volume)) + geom_line()  + ylab("Daily pay chat volume") +xlab("")

ggplot(new_df, aes(week_num_of_year, Contact_Volume)) + geom_line()  + ylab("Daily pay chat volume") +xlab("")

ggplot(new_df, aes(month_num, Contact_Volume)) + geom_line()  + ylab("Daily pay chat volume") +xlab("")


####################################################################################################################
head(new_df)
train_1 = aggr_new_df[aggr_new_df$Date >= "2016-01-01" , ]
train_2 = new_df[new_df$Date >= "2016-01-01",]

dim(train_1)
dim(train_2)

summary(train_1$Date)
summary(train_2$Date)

head(train_1)
head(train_2)
#===============================================================================================
#Fitting ARIMA on daily data
#==============================

daily_data = train_1

#Fitting ARIMA on Hourly data
#==============================

Hourly_data = train_2


#####################################################################################################################
#####################################################################################################################

Hourly_data = train_2

dim(Hourly_data)

H_train = Hourly_data[Hourly_data$Date < "2018-07-01",]
H_test = Hourly_data[Hourly_data$Date >= "2018-07-01",]

dim(H_train)
dim(H_test)

head(H_train)
head(H_test)


plot(ts(H_train[,c(4)], start=c(year(H_train[1,1]),1), frequency= 24*365.25))

count_ts = ts(H_train[, c('Contact_Volume')])
H_train$clean_cnt = tsclean(count_ts)
head(H_train)

plot(ts(H_train[,c(10)], start=c(year(H_train[1,1]),1), frequency= 24*365.25))



Station.ts <- ts(H_train[,c(10)], start=c(year(H_train[1,1]),1), frequency= 24*365.25)
autoplot(Station.ts)
plot(Station.ts)

start(Station.ts)
end(Station.ts)
head(Station.ts,30)


adf.test((Station.ts), alternative = "stationary")
kpss.test((Station.ts), null = c("Level"))
kpss.test((Station.ts), null = c("Trend"))


Acf(Station.ts, main='',lag.max = 500)
Pacf(Station.ts, main='',lag.max = 500)

Acf(Station.ts, main='',lag.max = 12000)
Pacf(Station.ts, main='',lag.max = 12000)


decomp_stl = stl(Station.ts, s.window="periodic")
plot(decomp_stl)
autoplot(decomp_stl) #,facet=TRUE)


msts_obj = msts(na.omit(Station.ts),seasonal.periods=c(24, 24*7, 24*365.25),start=c(2016,1))
decomp_mstl = mstl(msts_obj) #, s.window="periodic")
autoplot(decomp_mstl,facet=TRUE)


deseasonal_obj <- seasadj(decomp_stl)

adf.test((deseasonal_obj), alternative = "stationary")
kpss.test((deseasonal_obj), null = c("Level"))
kpss.test((deseasonal_obj), null = c("Trend"))


Acf(deseasonal_obj, main='',lag.max = 120)
Pacf(deseasonal_obj, main='',lag.max = 120)



count_d1 = diff(deseasonal_obj, differences = 2)
plot(count_d1)
autoplot(count_d1)

adf.test((count_d1), alternative = "stationary")
kpss.test((count_d1), null = c("Level"))
kpss.test((count_d1), null = c("Trend"))


Acf(count_d1, main='',lag.max = 500)
Pacf(count_d1, main='',lag.max = 100)


autoplot(count_d1,Facet = TRUE)

library(TSA)
p <- periodogram(Station.ts)  #we look at the possible frequencies of seasonality
#now we try to figure out which they are
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top5 = head(order, 5)
# display the 5 highest "power" frequencies
top5
# convert frequency to time periods
time <- 1/top5$f
time



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#==============================================
#exponential smoothing methods
#-==================
Hourly_data = train_2

H_train = Hourly_data[Hourly_data$Date < "2018-07-01",]
H_test = Hourly_data[Hourly_data$Date >= "2018-07-01",]


summary(H_train$Contact_Volume)
H_train$Contact_Volume[H_train$Contact_Volume==0] = 1
summary(H_train$Contact_Volume)
quantile(H_train$Contact_Volume,c(seq(.05,1,.05)),na.rm = F)
quantile(H_train$clean_cnt,c(seq(.05,1,.05)),na.rm = F)


#Station.ts <- ts(H_train[,c(10)], start=c(year(H_train[1,1]),1), frequency= 24)

Station.ts <- ts(H_train[,c(10)], start=c(week(H_train[1,1]),1), frequency= 24)


summary(Station.ts)
quantile(Station.ts,c(seq(.05,1,.05)),na.rm = F)

start(Station.ts)
end(Station.ts)
head(Station.ts)
length(H_train$Contact_Volume)
length(Station.ts)
autoplot(Station.ts)


#--------------------------------------------------------
fit1 <- ets(Station.ts,model = ("AAA"),damped = T )
summary(fit1)
autoplot(fit1)

fc1 = forecast(fit1,h=744)
plot(fc1)

summary(fc1$mean)

autoplot(Station.ts) +
  autolayer(fc1, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))


cbind('Residuals' = residuals(fit1),
      'Forecast errors' = residuals(fit1,type='response')) %>%
  autoplot(facet=TRUE) + xlab("Year") + ylab("")

#--------------------------------------------------------
fit2 <- ets(Station.ts,model = ("AAM"),damped = F )
summary(fit2)
autoplot(fit2)

fc2 = forecast(fit2,h=744)
plot(fc2)

summary(fc2$mean)

autoplot(Station.ts) +
  autolayer(fc2, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))

#--------------------------------------------------------
fit3 <- ets(Station.ts,model = ("MAA"),damped = T )
summary(fit3)
autoplot(fit3)

fc3 = forecast(fit3,h=744)
plot(fc3)

autoplot(Station.ts) +
  autolayer(fc3, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))

#--------------------------------------------------------
fit4 <- ets(Station.ts,model = ("MAM"),damped = T )
summary(fit4)
autoplot(fit4)

fc4 = forecast(fit4,h=744)
plot(fc4)

autoplot(Station.ts) +
  autolayer(fc4, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))


autoplot(Station.ts) +
  autolayer(fc1, series="HW multi damped1", PI=FALSE)+
  autolayer(fc3, series="HW multi damped3", PI=FALSE)+
  autolayer(fc4, series="HW multi damped4", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))


cbind('Residuals' = residuals(fit),
      'Forecast errors' = residuals(fit,type='response')) %>%
  autoplot(facet=TRUE) + xlab("Year") + ylab("")


#--------------------------------------------------------
fit5 <- ets(Station.ts,model = ("ANA"),damped = F )
summary(fit5)
autoplot(fit4)

fc5 = forecast(fit5,h=744)
plot(fc5)

autoplot(Station.ts) +
  autolayer(fc4, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))


#--------------------------------------------------------
fit6 <- ets(Station.ts,model = ("MNA"),damped = F)
summary(fit6)
autoplot(fit6)

fc6 = forecast(fit6,h=744)
plot(fc6)

autoplot(Station.ts) +
  autolayer(fc4, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))


#--------------------------------------------------------
fit7 <- ets(Station.ts,model = ("MNM"),damped = F )
summary(fit7)
autoplot(fit7)

fc7 = forecast(fit7,h=744)
plot(fc7)
summary(fc7$mean)

autoplot(Station.ts) +
  autolayer(fc7, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="hourly forecasts"))


#================================================================================================

Evalution = data.frame(Hourly_data[Hourly_data$Date >= "2018-07-01",c(2,3,4,5,7)])
head(Evalution)
Evalution$Predicted_Volume = (fc7$mean)
head(Evalution)

#dput(names(Evalution))
names(Evalution) = c("Date", "Hour_of_Day", "Actual_Volume", "day_of_week","week_num_of_mnth", "Predicted_Volume")
Evalution$Predicted_Volume = round(Evalution$Predicted_Volume)
head(Evalution)

Evalution$error = Evalution$Actual_Volume-Evalution$Predicted_Volume
Evalution$error_percent = round((Evalution$error/(if_else(Evalution$Actual_Volume==0,1,Evalution$Actual_Volume)))*100,0)

summary(Evalution$error_percent)

quantile(Evalution$error_percent,c(seq(.05,1,.05)),na.rm = F)
quantile(Evalution$error_percent,c(seq(.02,1,.02)),na.rm = F)

quantile(Evalution$error_percent[Evalution$week_num_of_mnth==1],c(seq(.05,1,.05)),na.rm = F)
quantile(Evalution$error_percent[Evalution$week_num_of_mnth==2],c(seq(.02,1,.02)),na.rm = F)
quantile(Evalution$error_percent[Evalution$week_num_of_mnth==3],c(seq(.02,1,.02)),na.rm = F)
quantile(Evalution$error_percent[Evalution$week_num_of_mnth==4],c(seq(.02,1,.02)),na.rm = F)
quantile(Evalution$error_percent[Evalution$week_num_of_mnth==5],c(seq(.02,1,.02)),na.rm = F)



##############################################################################################################

#----------------------------------------------------------------------------------------------
#https://robjhyndman.com/hyndsight/dailydata/

#BATS and TBATS are an extension of ETS.
#-------------------------------------------
#When the time series is long enough to take in more than a year, then it may be necessary to allow 
#for annual seasonality as well as weekly seasonality. In that case, a multiple seasonal model such 
#as TBATS is required.


count_ma = msts(na.omit(Station.ts),seasonal.periods=c(24, 24*7, 24*365.25),start=c(2016,7,1))
decomp = mstl(count_ma) #, s.window="periodic")

plot(mstl(decomp))
autoplot(decomp,facet=TRUE)

mstl(count_ma) %>% autoplot(facet=TRUE)
mstl(count_ma, lambda='auto') %>% autoplot(facet=TRUE)

#deseasonal_cnt <- seasadj(decomp) #to deseasonalise data

y <- msts(na.omit(Station.ts),seasonal.periods=c(24, 24*7, 24*365.25),start=c(2016,7,1))
fit <- tbats(y)
fc <- forecast(fit, h=24*31)
plot(fc)



##############################################################################################################
#----------------------------------------
#so Fourier terms can be used to model the annual seasonality. Suppose we use K=5 Fourier terms 
#to model annual seasonality, and that the holiday dummy variables are in the vector holiday with
#100 future values in holidayf

#use a regression model with ARIMA errors, where the regression terms include any dummy holiday 
#effects as well as the longer annual seasonality

y <- ts(x, frequency=7)
z <- fourier(ts(x, frequency=365.25), K=5)
zf <- fourier(ts(x, frequency=365.25), K=5, h=100)
fit <- auto.arima(y, xreg=cbind(z,holiday), seasonal=FALSE)
fc <- forecast(fit, xreg=cbind(zf,holidayf), h=100)

#or 

fit <- Arima(y, order=c(2,0,1), xreg=fourier(y, K=4))
plot(forecast(fit, h=2*m, xreg=fourier(y, K=4, h=744)))

#----------------------------------------------------------------------------------------------
#Alternatively (and the only easy option if there are missing data) is to use Fourier terms for the 
#seasonal periods and ARMA errors to handle any remaining serial correlation. The ARIMA functions in
#R do not automatically handle multiple seasonal periods, but the following R code should work:


class(x)
head(x)
tail(x)
start(x)
end(x)
frequency(x)

x <- ts(Station.ts, frequency=24)
seas1 <- fourier(x, K=3)
seas2 <- fourier(ts(x, freq=24*7), K=3)
fit <- auto.arima(Station.ts, xreg=cbind(seas1,seas2))

seas1.f <- fourierf(x, K=3, h= 24*31)
seas2.f <- fourierf(ts(x, freq=24*7), K=3, h= 24*31)

fc <- forecast(fit, xreg=cbind(seas1.f,seas2.f))

plot(fc)

#---------------------------------------------------------------------------------------------



