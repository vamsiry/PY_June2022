
#R will report the value of the log likelihood of the data; that is, the logarithm of the probability
#of the observed data coming from the estimated model. For given values of p, d and q, R will try to
#maximise the log likelihood when finding parameter estimates.

#sudo systemctl start rstudio-server
#-------------------------------------

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

head(df)
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
new_df$Contact_Volume[is.na(new_df$Contact_Volume)] = 1
new_df$zero_vloume = NULL
summary(new_df)


new_df_1 = new_df

head(new_df)
new_df = new_df[,c(3,1,2,4)]
head(new_df)

#===================================================================================================
#===================================================================================================
#===================================================================================================
train_2 = new_df[new_df$Date >= "2016-07-01",]

Hourly_data = train_2

adf.test(Hourly_data$Contact_Volume, alternative = "stationary")

kpss.test(Hourly_data$Contact_Volume, null = c("Level"))

kpss.test(Hourly_data$Contact_Volume, null = c("Trend"))


#####################################################################################################################

#Station.ts <- ts(Hourly_data[,c(4)], start=c(year(Hourly_data[1,1]),1), frequency= 24*365.25)


#for identifying  multiple seasonality : 
#----------------------------------------------------
library(TSA)
p <- periodogram(Station.ts)  #we look at the possible frequencies of seasonality
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top5 = head(order, 5)
# display the 5 highest "power" frequencies
top5
# convert frequency to time periods
time <- 1/top5$f
time

#####################################################################################################################

#Detailed Metrics
#------------------
plot.ts(Station.ts)
abline(reg=lm(Station.ts~time(Station.ts)))


#smoothing/cleaning data
#-----------------------
#R provides a convenient method for removing time series outliers: tsclean() as part of its forecast package. tsclean() 
#identifies and replaces outliers using series smoothing and decomposition.

count_ts = ts(Hourly_data[, c('Contact_Volume')])
Hourly_data$clean_cnt = tsclean(count_ts)
head(Hourly_data)


ggplot() + geom_line(data = Hourly_data, aes(x = Date, y = Contact_Volume)) + ylab('Actual volume')

ggplot() + geom_line(data = Hourly_data, aes(x = Date, y = clean_cnt)) + ylab('Smoothed volume')


#Decompose Your Data
#------------------------------
count_ma = ts(na.omit(Hourly_data$Contact_Volume), frequency=24)
decomp = stl(count_ma, s.window="periodic")
deseasonal_cnt <- seasadj(decomp)
plot(decomp)


class(Hourly_data)
class(count_ma)

#Stationarity(checking stationarity assumption)
#--------------------------------------------------------
adf.test(count_ma, alternative = "stationary")
kpss.test(count_ma, null = c("Level"))
kpss.test(count_ma, null = c("Trend"))


adf.test(diff((deseasonal_cnt)), alternative = "stationary")
kpss.test(diff((deseasonal_cnt)), null = c("Level"))
kpss.test(diff((deseasonal_cnt)), null = c("Trend"))



adf.test(diff((count_ma)), alternative = "stationary")
kpss.test(diff((count_ma)), null = c("Level"))



#Autocorrelations and Choosing Model Order
#----------------------------------------------------
Acf(count_ma, main='')

Pacf(count_ma, main='')

count_d1 = diff(deseasonal_cnt, differences = 1)

plot(count_d1)
Acf(count_d1, main='')
Pacf(count_d1, main='')

count_d2 = diff(deseasonal_cnt, differences = 24)

Acf(count_d2, main='')
Pacf(count_d2, main='')



adf.test((count_d1), alternative = "stationary")
kpss.test((count_d1), null = c("Level"))
kpss.test((count_d1), null = c("Trend"))



length(deseasonal_cnt)
head(deseasonal_cnt)
tail(deseasonal_cnt)
length(deseasonal_cnt[-c(17521:18264)])


#####################################################################################################################
#Fitting ARIMA on Hourly data

#length(deseasonal_cnt)
#length(deseasonal_cnt) - 31*24

#include.drift = TRUE,

#Non seasonal model

fit_1<- forecast::auto.arima((deseasonal_cnt[-c(17521:18264)]),stepwise=FALSE,
                             approximation=FALSE,seasonal = FALSE,trace = TRUE, lambda="auto")
                             
fit_1
tsdisplay(residuals(fit_1), lag.max=45, main='(0,1,5) Model Residuals')


fcast_1 <- forecast(fit_1, h=744)
plot(fcast_1)
lines(ts(deseasonal_cnt))

accuracy(fcast_1, Evalution$Actual_Volume)["Test set","RMSE"]

accuracy(fcast_1, Evalution$Actual_Volume)

res <- residuals(fcast_1)
autoplot(res)
ggAcf(res,lag.max = 500) + ggtitle("ACF of residuals")


checkresiduals(fcast_1)
confint(fcast_1)

autoplot(fcast_1, series="fc") +
  autolayer(ts(count_ma), series="ordered data") 


#-----
Evalution = data.frame(Hourly_data[Hourly_data$Date >= "2018-07-01",c(2,3,4,5)])
head(Evalution)
Evalution$Predicted_Volume = (fcast_1$mean)

names(Evalution) = c("Date", "Hour_of_Day", "Actual_Volume", "day_of_week", "Predicted_Volume")
Evalution$Predicted_Volume = round(Evalution$Predicted_Volume,0)
head(Evalution)

Evalution$error = Evalution$Actual_Volume-Evalution$Predicted_Volume
Evalution$error_percent = round((Evalution$error/(if_else(Evalution$Actual_Volume==0,1,Evalution$Actual_Volume)))*100,0)

summary(Evalution$error_percent)
quantile(Evalution$error_percent,c(seq(.02,1,.02)),na.rm = F)


#===========================================================================================================
#fit_2.1 <- forecast::auto.arima(ts(count_ma[-c(17521:18264)],frequency = 24),stepwise=FALSE,
#                              approximation=FALSE,seasonal = TRUE,trace = TRUE, lambda="auto")

fit_2.1 <- forecast::auto.arima(ts(count_ma[-c(17521:18264)],frequency = 24),stepwise=FALSE,
                                approximation=FALSE,seasonal = TRUE,trace = TRUE, lambda="auto")


fit_2.1
tsdisplay(residuals(fit_2.1), lag.max=45, main='(0,1,5) Model Residuals')

fcast_2 <- forecast(fit_2.1, h=744)
plot(fcast_2)
lines(ts(deseasonal_cnt))

#-----
Evalution = data.frame(Hourly_data[Hourly_data$Date >= "2018-07-01",c(2,3,4,5)])
head(Evalution)
Evalution$Predicted_Volume = (fcast_2$mean)

names(Evalution) = c("Date", "Hour_of_Day", "Actual_Volume", "day_of_week", "Predicted_Volume")
Evalution$Predicted_Volume = round(Evalution$Predicted_Volume,0)

Evalution$error = Evalution$Actual_Volume-Evalution$Predicted_Volume
Evalution$error_percent = round((Evalution$error/(if_else(Evalution$Actual_Volume==0,1,Evalution$Actual_Volume)))*100,0)

summary(Evalution$error_percent)
quantile(Evalution$error_percent,c(seq(.02,1,.02)),na.rm = F)


#===========================================================================================================
#ARIMA(1,0,2)(2,1,0)[24] with drift         : 51222.62

fit_2 = Arima(ts(count_ma[-c(17521:18264)],frequency = 24), order=c(3,1,1), seasonal=c(1,0,0),
              lambda = "auto", include.drift = TRUE,
              include.mean = TRUE,include.constant = TRUE,biasadj = TRUE)


fit_2 = Arima(ts(count_ma[-c(17521:18264)],frequency = 24), order=c(1,0,2), seasonal=c(2,1,0),
              lambda = "auto", include.drift = FALSE,
              include.mean = TRUE,include.constant = TRUE,biasadj = TRUE)



fit_2
tsdisplay(residuals(fit_2), lag.max=45, main='(0,1,5) Model Residuals')

fcast_2 <- forecast(fit_2, h=744)
plot(fcast_2)
lines(ts(count_ma))

accuracy(fcast_2, Evalution$Actual_Volume)["Test set","RMSE"]



#-----
Evalution = data.frame(Hourly_data[Hourly_data$Date >= "2018-07-01",c(2,3,4,5)])
head(Evalution)
Evalution$Predicted_Volume = (fcast_2$mean)

names(Evalution) = c("Date", "Hour_of_Day", "Actual_Volume", "day_of_week", "Predicted_Volume")
Evalution$Predicted_Volume = round(Evalution$Predicted_Volume,0)

Evalution$error = Evalution$Actual_Volume-Evalution$Predicted_Volume
Evalution$error_percent = round((Evalution$error/(if_else(Evalution$Actual_Volume==0,1,Evalution$Actual_Volume)))*100,0)

summary(Evalution$error_percent)
quantile(Evalution$error_percent,c(seq(.02,1,.02)),na.rm = F)



#=========================================================================================================
xx = ts(Hourly_data[, c('Contact_Volume')])

xx %>% autoplot()

xx %>% diff(lag=365.25*24) %>% ggtsdisplay()

xx %>% diff(lag=365.25*24) %>% diff(lag = 2) %>% ggtsdisplay()

xx %>% Arima(order=c(2,1,0), seasonal=c(1,1,0)) %>% residuals() %>% ggtsdisplay()



#######################################################################################################
#ARIMA(1,0,2)(2,1,0)[24] with drift : 51222.62

#use a regression model with ARIMA errors, where the regression terms include any dummy holiday 
#effects as well as the longer annual seasonality

#fit_2 <- forecast::auto.arima(ts(count_ma[-c(17521:18264)],frequency = 24),stepwise=FALSE,D=1,
#                              approximation=FALSE,seasonal = TRUE,trace = TRUE, lambda="auto")

y <- ts(count_ma[-c(17521:18264)],frequency = 24)
z <- fourier(ts(count_ma[-c(17521:18264)],frequency = 24), K=5)
zf <- fourier(ts(count_ma[-c(17521:18264)],frequency = 24), K=5, h=744)


#fit <- auto.arima(y, xreg= z, seasonal=TRUE, stepwise=FALSE, 
#                  approximation=FALSE,trace = TRUE, lambda="auto")

#fit <- auto.arima(y, xreg= z, seasonal=TRUE,stepwise=FALSE, 
#                  approximation=FALSE,trace = TRUE, lambda="auto",allowdrift = TRUE)


#ARIMA(1,0,2)(2,1,0)[24] with drift : 51222.62
fit = Arima(y, xreg = z, order=c(3,1,1), seasonal=c(1,0,0),lambda = "auto", include.drift = TRUE,
            include.mean = TRUE,include.constant = TRUE,biasadj = TRUE)

fit

fc <- forecast(fit, xreg= zf, h=744)

tsdisplay(residuals(fit), lag.max=45, main='ARIMA(3,1,1)(1,0,0)[24] errors')


fitt = Arima(y, xreg = z, order=c(6,1,3), seasonal=c(0,0,0))

fitt

fcc <- forecast(fitt, xreg= zf, h=744)

accuracy(fcc$mean, Evalution$Actual_Volume)["Test set","RMSE"]

#or 
#fit <- Arima(y, order=c(2,0,1), xreg=fourier(y, K=4))
#plot(forecast(fit, h=744, xreg=fourier(y, K=4, h=744)))

#-----
Evalution = data.frame(Hourly_data[Hourly_data$Date >= "2018-07-01",c(2,3,4,5)])
head(Evalution)
Evalution$Predicted_Volume = (fc$mean)

names(Evalution) = c("Date", "Hour_of_Day", "Actual_Volume", "day_of_week", "Predicted_Volume")
Evalution$Predicted_Volume = round(Evalution$Predicted_Volume,0)

Evalution$error = Evalution$Actual_Volume-Evalution$Predicted_Volume
Evalution$error_percent = round((Evalution$error/(if_else(Evalution$Actual_Volume==0,1,Evalution$Actual_Volume)))*100,0)

summary(Evalution$error_percent)
quantile(Evalution$error_percent,c(seq(.02,1,.02)),na.rm = F)



#----------------------------------------------------------------------------------------------
#Alternatively (and the only easy option if there are missing data) is to use Fourier terms for the 
#seasonal periods and ARMA errors to handle any remaining serial correlation. The ARIMA functions in
#R do not automatically handle multiple seasonal periods, but the following R code should work:

#https://content.pivotal.io/blog/forecasting-time-series-data-with-multiple-seasonal-periods


#fit <- auto.arima(y, xreg= z, seasonal=TRUE,stepwise=FALSE, 
#                  approximation=FALSE,trace = TRUE, lambda="auto",allowdrift = TRUE)


y <- msts(y, c(24,24*7)) # multiseasonal ts

aic_vals_temp = NULL
aic_vals = NULL

for (i in 1:5){
  for (j in 1:12){
    
    fit =  auto.arima(y, xreg=fourier(y, K=c(i,j)), seasonal=F, D = 0,max.p = 0,max.Q = 0,stepwise=FALSE,
                      approximation=FALSE,trace = TRUE, lambda="auto",allowdrift = TRUE)
    
    fit_f <- forecast(fit, xreg= fourier(y, K=c(i,j), 744), 744)
    
    accuraci = accuracy(fit_f, Evalution$Actual_Volume)["Test set","RMSE"]
    
    aic_vals_temp = cbind(i,j,fit$aic,accuraci)
    
    aic_vals = rbind(aic_vals,aic_vals_temp)
    
  }
  
}


colnames(aic_vals) = c("Fourier_24" ,"Fourier_168" ,"AIC_value","accuraci")
aic_vals = data.frame(aic_vals)
aic_vals
minAICval = min(aic_vals$AIC_value)
minRMSE = min(aic_vals$accuraci)
minvals = aic_vals[aic_vals$AIC_value == minAICval,]
minvals

aic_vals[aic_vals$accuraci == minRMSE,]

#aic_vals2 = aic_vals
aic_vals2
aic_vals3

aic_vals23 = rbind(aic_vals2,aic_vals3)

#----------------------------------------------------------------------------------------

y <- ts(count_ma[-c(17521:18264)],frequency = 24)

y <- msts(y, c(24,24*7)) # multiseasonal ts

fit =  auto.arima(y, xreg=fourier(y, K=c(5,5)), seasonal=T, D = 0,max.p = 10,max.q = 10,
                  max.P = 10,max.Q = 10,stepwise=FALSE,max.d = 2,max.D = 1,max.order = 1,
                  biasadj = TRUE,
                  approximation=FALSE,trace = TRUE, lambda="auto",allowdrift = TRUE)



y <- msts(y, c(24,24*7)) # multiseasonal ts

aic_vals_temp = NULL
aic_vals = NULL



for (i in 0:6){
  for (j in 0:2){
    for (k in 0:6){
      for (L in 0:3){
        for( M in 0:1){
          for (N in 0:3){
            fit =  auto.arima(y, xreg=fourier(y, K=c(4,5)),
                              start.p = i,d = j,start.q = k,max.p = i,max.d = j,max.q = k,
                              max.P = L,max.D = M,max.Q = N,start.P = L,D = M,start.Q = N,
                              seasonal=TRUE, stepwise=FALSE,trace = TRUE,
                              lambda="auto",biasadj = TRUE,approximation = FALSE)
            
            #fit = Arima(y,xreg = fourier(y, K=c(4,5)), order=c(i,j,k), seasonal=c(L,M,N),
            #            lambda = "auto",biasadj = TRUE,method="CSS-ML",)
            
            fit_f <- forecast(fit, xreg= fourier(y, K=c(4,5), 744), 744)
            
            accuraci = accuracy(fit_f$mean, Evalution$Actual_Volume,SIMPLIFY = FALSE)
            
            aic_vals_temp = cbind(i,j,k,L,M,N,fit$aic,accuraci[1],accuraci[2],accuraci[3],accuraci[4],accuraci[5])
            
            aic_vals = rbind(aic_vals,aic_vals_temp)
            
          }
        }
      }
    }
  }
}





#colnames(aic_vals) = c("Fourier_24" ,"Fourier_168" ,"AIC_value","ME","RMSE","MAE","MPE","MAPE")

colnames(aic_vals) = c("i" ,"j" ,"k","L","M","N","AIC_value","ME","RMSE","MAE","MPE","MAPE")
aic_vals = data.frame(aic_vals)
aic_vals
minAICval = min(aic_vals$AIC_value)
minRMSE = min(aic_vals$accuraci)
minvals = aic_vals[aic_vals$AIC_value == minAICval,]
minvals

?Arima

#-------------------------------------------------------------------------------------------------

gastbats <- tbats(y)
gastbats
fc2 <- forecast(gastbats, h=744)
plot(fc2, ylab="volumes per day")
lines(ts(count_ma))

#-----
Evalution = data.frame(Hourly_data[Hourly_data$Date >= "2018-07-01",c(2,3,4,5)])
head(Evalution)
Evalution$Predicted_Volume = (fc2$mean)

names(Evalution) = c("Date", "Hour_of_Day", "Actual_Volume", "day_of_week", "Predicted_Volume")
Evalution$Predicted_Volume = round(Evalution$Predicted_Volume,0)

Evalution$error = Evalution$Actual_Volume-Evalution$Predicted_Volume
Evalution$error_percent = round((Evalution$error/(if_else(Evalution$Actual_Volume==0,1,Evalution$Actual_Volume)))*100,0)

head(Evalution)
summary(Evalution$error_percent)
quantile(Evalution$error_percent,c(seq(.02,1,.02)),na.rm = F)





#============================================================================================

#---------------------------------------------------------------------------------------------

#https://stackoverflow.com/questions/4479463/using-fourier-analysis-for-time-series-prediction


#to find "patterns". I assume that means finding the dominant frequency components in the 
#observed data. Then yes, take the Fourier transform, preserve the largest coefficients, 
#and eliminate the rest.


#P.S. Locally Stationary Wavelet may be better than fourier extrapolation. LSW is 
#commonly used in predicting time series. The main disadvantage of fourier extrapolation 
#is that it just repeats your series with period N, where N - length of your time series


#https://en.wikipedia.org/wiki/Fourier_analysis


#Good topic : http://blog.supplyframe.com/2014/03/05/how-to-identify-patterns-in-time-series-data-part-i-discrete-fourier-transform/

#Good read : http://astrostatistics.psu.edu/su10/lectures/BrazilTSA.pdf

# https://www.researchgate.net/post/Can_anyone_help_me_with_wavelet_time_series_analysis

#wavelet methods in statistics with r pdf


#Discrete Fourier Transform (DFT)
#Fast Fourier Transform (FFT)



#------------------------------------------------------------------------------------------------

h <-744

ETS <- forecast(ets(y), h=h)
ARIMA <- forecast(auto.arima(y,  biasadj=TRUE),h=h) #lambda=0,
STL <- stlf(y,  h=h, biasadj=TRUE) #lambda=0,
NNAR <- forecast(nnetar(y), h=h)
TBATS <- forecast(tbats(y, biasadj=TRUE), h=h)

Combination <- (ETS[["mean"]] + ARIMA[["mean"]] +
                  STL[["mean"]] + NNAR[["mean"]] + TBATS[["mean"]])/5



autoplot(y) +
  autolayer(ETS, series="ETS", PI=FALSE) +
  autolayer(ARIMA, series="ARIMA", PI=FALSE) +
  autolayer(STL, series="STL", PI=FALSE) +
  autolayer(NNAR, series="NNAR", PI=FALSE) +
  autolayer(TBATS, series="TBATS", PI=FALSE) +
  autolayer(Combination, series="Combination") +
  xlab("Year") + ylab("$ billion") +
  ggtitle("Australian monthly expenditure on eating out")




c(ETS = accuracy(ETS, Evalution$Actual_Volume)["Test set","RMSE"],
  ARIMA = accuracy(ARIMA, Evalution$Actual_Volume)["Test set","RMSE"],
  `STL-ETS` = accuracy(STL, Evalution$Actual_Volume)["Test set","RMSE"],
  NNAR = accuracy(NNAR, Evalution$Actual_Volume)["Test set","RMSE"],
  TBATS = accuracy(TBATS, Evalution$Actual_Volume)["Test set","RMSE"],
  Combination = accuracy(Combination, Evalution$Actual_Volume)["Test set","RMSE"])



Evalution = data.frame(Hourly_data[Hourly_data$Date >= "2018-07-01",c(2,3,4,5)])
head(Evalution)
Evalution$Predicted_Volume = (ETS$mean)

names(Evalution) = c("Date", "Hour_of_Day", "Actual_Volume", "day_of_week", "Predicted_Volume")
Evalution$Predicted_Volume = round(Evalution$Predicted_Volume,0)

Evalution$error = Evalution$Actual_Volume-Evalution$Predicted_Volume
Evalution$error_percent = round((Evalution$error/(if_else(Evalution$Actual_Volume==0,1,Evalution$Actual_Volume)))*100,0)

head(Evalution)
summary(Evalution$error_percent)
quantile(Evalution$error_percent,c(seq(.02,1,.02)),na.rm = F)

#====================================================================================================




