# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:52:05 2022

@author: rvamsikrishna
"""




import numpy as np
import pandas as pd

df = pd.DataFrame()

#%%
df.infer_objects().dtypes #Attempt to infer better dtypes for object columns.
#%%
#to_datetime --  Convert argument to datetime.
#to_timedelta -- Convert argument to timedelta.
#to_numeric -- Convert argument to numeric type.



#%%
#%%
import datetime

df = pd.DataFrame({ "date":[datetime.date(2012,x,1) for x in range(1,11)], 
                     "returns":0.05*np.random.randn(10), 
                     "dummy":np.repeat(1,10) 
                      })
df    
#%%
#%%
#Range Data
#--------------
#pd.data_range(date,period,frequency): 

## Create date  --The last parameter is the frequency: 
#day: 'D,' month: 'M' and year: 'Y.'

#freq parameter options link below
#https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

dates_d = pd.date_range('20300101', periods=6, freq='D')
print(type(dates_d))
print(dates_d)
#%%
pd.date_range('20300101', periods=6, freq='M') #monthly data

#%%
pd.PeriodIndex(start ='2004-11-21 08:45:21 ', \
               end ='2004-11-21 8:45:29', freq ='S') 
#%%
#%%


#%%
#%%
data = {'Salary':[1000, 2222, 3321, 4414, 5151],
       'Name': ['Pete', 'Steve','Brian','Ryan', 'Jim'],
       'Share':[29.88, 19.05, 8.17,7.3, 6.15],
       'Date':['11/24/2020', '12/21/2019', '10/14/2018', '12/13/2017', '01/08/2017'],
       'Date2': [20120902, 20130413, 20140921, 20140321, 20140321]}

print(data)

import pandas as pd
df = pd.DataFrame(data)
df.head()

#%%
df.dtypes
#%%
df.info()
#%%
#format='%Y%m%d-%H%M%S' #fromat with timestamp
#format='%y%m%d' #Format with date
#%%
#1 Convert an Object (string) Column:
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes
#%%
#2 Convert an Integer Column:
df['Date2'] = pd.to_datetime(df['Date2']) #can convert object/string, integer,
df.dtypes
#%%
#Pandas Convert Column with the astype() Method
df['Date'] = df['Date'].astype('datetime64[ns]') ##can convert only string
df.dtypes
#%%
df['Date2'] = df['Date2'].astype('datetime64[ns]')
df.dtypes
#%%



#%%
#%%
#1. Basic Basic conversion with scalar string
string_to_convert = '2020-02-01'    
type(string_to_convert)

#%%
new_date = pd.to_datetime(string_to_convert)
print(type(new_date))

#%%
#2. Convert Pandas Series to datetime
s = pd.Series(['2020-02-01','2020-02-02','2020-02-03','2020-02-04'])
s
#%%
s = pd.to_datetime(s)
print(s)
print(type(s))
#%%
#3. Convert Pandas Series to datetime w/ custom format
s = pd.Series(['My 3date is 01199002','My 3date is 02199015',
           'My 3date is 03199020','My 3date is 09199204'])
s
#%%
s = pd.to_datetime(s, format="My 3date is %m%Y%d")
s
#%%
#4. Convert Unix integer (days) to datetime

#You can also convert integers into Datetimes. You'll need to keep 
#two things in mind

#1.Reference point = What time do you want to start 'counting' your units from?
#2.Unit = Is your integer in terms of # of days, seconds, years, etc.?

pd.to_datetime(14554, unit='D', origin='unix')

#%%
#5. Convert integer (seconds) to datetime

#More often, you'll have a unix timestamp that is expresses in seconds. 
#As in seconds away from the default origin of 1970-01-01.

#For example, at the time of this post, we are 1,600,355,888 seconds away 
#from 1970-01-01. That's lot of seconds!

pd.to_datetime(1600355888, unit='s', origin='unix')

#%%
#Bonus: 6. Change your origin or reference point

#Say your dataset only has # of days after a certain time,but no datetimes. 
#You could either add all of those days via a pd.Timedelta().

#Or you could convert them to datetimes with a different origin. 
#Let's check this out from 2020-02-01.
pd.to_datetime(160, unit='D', origin='2020-02-01')

#%%
#%%




#%%
#%%
#https://www.programiz.com/python-programming/datetime
import datetime
datetime_object = datetime.datetime.now()
print(datetime_object)
#%%
#Example 2: Get Current Date
#----------------------------
date_object = datetime.date.today()
print(date_object)

#%%
#What's inside datetime?
#We can use dir() function to get a list containing all attributes of a module.
print(dir(datetime))
#%%
#Commonly used classes in the datetime module are:
#date Class
#time Class
#datetime Class
#timedelta Class

#%%
#datetime.date Class
#***********************
#You can instantiate date objects from the date class. 
#A date object represents a date (year, month and day).

#Example 3: Date object to represent a date
#-------------------------------------------------
d = datetime.date(2019, 4, 13)
print(d)

#%%
from datetime import date
a = date(2019, 4, 13)
print(a)
#%%
#Example 4: Get current date
#---------------------------
from datetime import date
today = date.today()
print("Current date =", today)
#%%
#Example 5: Get date from a timestamp
#------------------------------------------
#We can also create date objects from a timestamp. A Unix timestamp is
# the number of seconds between a particular date and January 1, 1970 at UTC.
# You can convert a timestamp to date using fromtimestamp() method.

from datetime import date

timestamp = date.fromtimestamp(1326244364)
print("Date =", timestamp)

#%%
#Example 6: Print today's year, month and day
from datetime import date

# date object of today's date
today = date.today() 

print("Current year:", today.year)
print("Current month:", today.month)
print("Current day:", today.day)

#%%
#%%
#datetime.time
#*****************

#A time object instantiated from the time class represents the local time.

#Example 7: Time object to represent time-
#-----------------------------------------
from datetime import time

# time(hour = 0, minute = 0, second = 0)
a = time()
print("a =", a)

# time(hour, minute and second)
b = time(11, 34, 56)
print("b =", b)

# time(hour, minute and second)
c = time(hour = 11, minute = 34, second = 56)
print("c =", c)

# time(hour, minute, second, microsecond)
d = time(11, 34, 56, 234566)
print("d =", d)

#%%
#Example 8: Print hour, minute, second and microsecond
#--------------------------------------------------------
from datetime import time

a = time(11, 34, 56)

print("hour =", a.hour)
print("minute =", a.minute)
print("second =", a.second)
print("microsecond =", a.microsecond)

#%%
#%%
#datetime.datetime
#*********************

#Example 9: Python datetime object
#----------------------------------
from datetime import datetime

#datetime(year, month, day)
a = datetime(2018, 11, 28)
print(a)

# datetime(year, month, day, hour, minute, second, microsecond)
b = datetime(2017, 11, 28, 23, 55, 59, 342380)
print(b)

#The first three arguments year, month and day in the datetime() 
#constructor are mandatory.

#%%
#Example 10: Print year, month, hour, minute and timestamp
#----------------------------------------------------------

from datetime import datetime

a = datetime(2017, 11, 28, 23, 55, 59, 342380)
print("year =", a.year)
print("month =", a.month)
print("day =", a.day)
print("hour =", a.hour)
print("minute =", a.minute)
print("second =", a.second)
print("timestamp =", a.timestamp())

#%%
#datetime.timedelta
#**********************
#A timedelta object represents the difference between two dates or times.

#Example 11: Difference between two dates and times
#--------------------------------------------------------
from datetime import datetime, date

t1 = date(year = 2018, month = 7, day = 12)
t2 = date(year = 2017, month = 12, day = 23)
t3 = t1 - t2
print("t3 =", t3)

t4 = datetime(year = 2018, month = 7, day = 12, hour = 7, minute = 9, second = 33)
t5 = datetime(year = 2019, month = 6, day = 10, hour = 5, minute = 55, second = 13)
t6 = t5 - t4
print("t6 =", t6)

print("type of t3 =", type(t3)) 
print("type of t6 =", type(t6))  

#%%
#Example 12: Difference between two timedelta objects
#---------------------------------------------------

from datetime import timedelta

t1 = timedelta(weeks = 2, days = 5, hours = 1, seconds = 33)
t2 = timedelta(days = 4, hours = 11, minutes = 4, seconds = 54)
t3 = t1 - t2

print("t3 =", t3)

#%%
#Example 13: Printing negative timedelta object
#------------------------------------------------
from datetime import timedelta

t1 = timedelta(seconds = 33)
t2 = timedelta(seconds = 54)
t3 = t1 - t2

print("t3 =", t3)
print("t3 =", abs(t3))

#%%
#Example 14: Time duration in seconds
#--------------------------------------
from datetime import timedelta

t = timedelta(days = 5, hours = 1, seconds = 33, microseconds = 233423)
print("total seconds =", t.total_seconds())

#You can also find sum of two dates and times using + operator. 
#Also, you can multiply and divide a timedelta object by integers and floats.

#%%
#Python format datetime
#****************************

#The way date and time is represented may be different in different places,
# organizations etc. It's more common to use mm/dd/yyyy in the US,
# whereas dd/mm/yyyy is more common in the UK.

#Python has strftime() and strptime() methods to handle this.

#%%
#Python strftime() - datetime object to string

#The strftime() method is defined under classes date, datetime and time. 
#The method creates a formatted string from a given date, datetime 
#or time object.

#Example 15: Format date using strftime()
#--------------------------------------------
from datetime import datetime

# current date and time
now = datetime.now()
print(now)

#%%
t = now.strftime("%H:%M:%S")
print("time:", t)

#%%
s1 = now.strftime("%m/%d/%Y, %H:%M:%S")
# mm/dd/YY H:M:S format
print("s1:", s1)

#%%
s2 = now.strftime("%d/%m/%Y, %H:%M:%S")
# dd/mm/YY H:M:S format
print("s2:", s2)

#%%
now.strftime("%Y") # 4 digits year
#%%
now.strftime("%y") # 2 digits year
#%%
now.strftime("%M") #minute
#%%
now.strftime("%m") #month
#%%
now.strftime("%D") # complete date
#%%
now.strftime("%d") #day num from date
#%%
now.strftime("%H") #Hour
#%%
now.strftime("%h") #shhort month name in eng
#%%
now.strftime("%S") #Second
#%%
#%%
#Example 2: Creating string from a timestamp
#------------------------------------------------
from datetime import datetime
timestamp = 1528797322
#%%
date_time = datetime.fromtimestamp(timestamp)
print("Date time object:", date_time)
print(type(date_time))

#%%
d = date_time.strftime("%m/%d/%Y, %H:%M:%S")
print("Output 2:", d)
print(type(d))
	
#%%
d = date_time.strftime("%d %b, %Y")
print("Output 3:", d)
#%%
d = date_time.strftime("%d %B, %Y")
print("Output 4:", d)
#%%
d = date_time.strftime("%I%p")
print("Output 5:", d)
#%%
#For formating codes list
#https://www.programiz.com/python-programming/datetime/strftime

#%%
#%%
#Example 3: Locale's appropriate date and time
#--------------------------------------------------
from datetime import datetime
timestamp = 1528797322
date_time = datetime.fromtimestamp(timestamp)
print(date_time)
print(type(date_time))

#%%
d = date_time.strftime("%c")
print("Output 1:", d)	
#%%
d = date_time.strftime("%x")
print("Output 2:", d)
#%%
d = date_time.strftime("%X")
print("Output 3:", d)

#%%
#%%
#Example 16: strptime()--create a datetime object from a string 
#******************************************************************
from datetime import datetime

date_string = "21 June, 2018"
print("date_string =", date_string)

#%%
date_object = datetime.strptime(date_string, "%d %B, %Y")
print("date_object =", date_object)
print(type(date_object))

#%%
#The strptime() method takes two arguments:
#1. a string representing date and time
#2. format code equivalent to the first argument

#By the way, %d, %B and %Y format codes are used for day,
# month(full name) and year respectively.
#%%
#Example 2: string to datetime object
#--------------------------------------
from datetime import datetime

dt_string = "12/11/2018 09:15:32"

# Considering date is in dd/mm/yyyy format
dt_object1 = datetime.strptime(dt_string, "%d/%m/%Y %H:%M:%S")
print("dt_object1 =", dt_object1)

# Considering date is in mm/dd/yyyy format
dt_object2 = datetime.strptime(dt_string, "%m/%d/%Y %H:%M:%S")
print("dt_object2 =", dt_object2)

#%%
#format codes
#https://www.programiz.com/python-programming/datetime/strptime
#%%
#ValueError in strptime()
#If the string (first argument) and the format code (second argument) 
#passed to the strptime() doesn't match, you will get ValueError. 

from datetime import datetime

date_string = "12/11/2018"
date_object = datetime.strptime(date_string, "%d %m %Y")

print("date_object =", date_object)


#%%









#%%