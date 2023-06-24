# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:01:48 2022

@author: rvamsikrishna
"""

#https://medium.com/data-folks-indonesia/10-things-to-do-when-conducting-your-exploratory-data-analysis-eda-7e3b2dfbf812

#https://towardsdatascience.com/exploratory-data-analysis-eda-python-87178e35b14

#EDA in Python uses data visualization to draw meaningful patterns and insights. 
#EDA helps in  preparation of data sets for analysis by removing irregularities in the data.

#Based on the results of EDA, companies also make business decisions, 
#which can have repercussions later.

#If EDA is not done properly then it can hamper the further steps in the 
#machine learning model building process.

#If done well, it may improve the efficacy of everything we do next.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#%%
df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//Models//PY EDA Marketing Analysis//marketing_analysis.csv")

df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//Models//PY EDA Marketing Analysis//marketing_analysis.csv",skiprows = 1)

df = pd.read_csv(‘data/my_dataset.csv’, skiprows=1, low_memory=False)

#%%
df.shape # num of Rows & Columns

#%%
df.head()

#%%
df.info() # each data type of columns and missing values

#%%
print(df.dtypes) #print data type of all columns

#%%
#df.describe() # Summary Statistics

#df.describe(include = ['object','int'] ) # Summary Statistics
#df.describe(exclude = ['object','int'] ) # Summary Statistics
#df.describe(percentiles = [.75,.85,.95] ) # Summary Statistics
#df.age.describe(percentiles = [.20,.40,.60,.80,.90] ) # Summary Statistics

#%%
df.describe().plot()
#%%
df.response.value_counts() # target class distribution

#%%
print(df.isnull().sum()) #column wise missing values

#%%
df = df.drop([‘url’],axis=1) #removing unnecessary URL column

#%%
X = df.iloc[:, :-1].values 

#That first colon (:) means that we want to take all the lines in our dataset.
# : -1 means that we want to take all of the columns of data except the last one.
# The .values on the end means that we want all of the values.

y = df.iloc[:, -1].values # tarfet column

#%%
#drop columns that have lots of missing values.
thresh = len(df) * 0.6
df.dropna(axis=1, thresh=thresh, inplace=True)
    
#%%
#mode imputation for categorical column (index[0] is high freq value in that column)
df.marital.fillna(df['marital'].value_counts().index[0])

#%%
#mean imputation
df.age.fillna(df['age'].mean())

#%%
#(2) Combining DataFrames
#--------------------
#Merging
#------------
#Inner join pandas data frame
inner_join = left_df.merge(right=right_df, how='inner', on='join_keys')

left_join = left_df.merge(right=right_df, how='left', on='join_keys')

right_join = left_df.merge(right=right_df, how='right', on='join_keys')

outer_join = left_df.merge(right=right_df, how='outer', on='join_keys', indicator=True)

#%%
#Concatenating
#------------------
# Vertical concat
pd.concat([october_df, november_df, december_df], axis=0)

# Horizontal concat
pd.concat([features_1to5_df, features_6to10_df, features_11to15_df], axis=1)

#%%
#(4) Working with time data
#--------------------------------
#Pandas comes with a function called to_datetime() that can 
#compress and convert multiple DataFrame columns into a 
#single Datetime object.

from itertools import product
import pandas as pd
import numpy as np

col_names = ["Day", "Month", "Year"]

df = pd.DataFrame(list(product([10, 11, 12], [8, 9], [2018, 2019])),
                   columns=col_names)

df['data'] = np.random.randn(len(df))

df = df.sort_values(['Year', 'Month'], ascending=[True, True])

print(df)
#%%
df.insert(loc=0, column="date", value=pd.to_datetime(df[col_names]))

df = df.drop(col_names, axis=1).squeeze()

print(df)

#%%
#(5) Mapping Items into Groups
#----------------------------------
import pandas as pd

foods = pd.Series(["Bread", "Rice", "Steak", "Ham", "Chicken",
                       "Apples", "Potatoes", "Mangoes", "Fish",
                       "Bread", "Rice", "Steak", "Ham", "Chicken",
                       "Apples", "Potatoes", "Mangoes", "Fish",
                       "Apples", "Potatoes", "Mangoes", "Fish",
                       "Apples", "Potatoes", "Mangoes", "Fish",
                       "Bread", "Rice", "Steak", "Ham", "Chicken",
                       "Bread", "Rice", "Steak", "Ham", "Chicken",
                       "Bread", "Rice", "Steak", "Ham", "Chicken",
                       "Apples", "Potatoes", "Mangoes", "Fish",
                       "Apples", "Potatoes", "Mangoes", "Fish",
                       "Apples", "Potatoes", "Mangoes", "Fish",
                       "Bread", "Rice", "Steak", "Ham", "Chicken",
                       "Bread", "Rice", "Steak", "Ham", "Chicken",])

groups_dict = {
    "Protein": ["Steak", "Ham", "Chicken", "Fish"],
    "Carbs": ["Bread", "Rice", "Apples", "Potatoes", "Mangoes"]
}

def membership_map(pandas_series, groups_dict):
    groups = {x: k for k, v in groups_dict.items() for x in v}
    mapped_series = pandas_series.map(groups)
    return mapped_series
    
mapped_data = membership_map(foods, groups_dict)
print(list(mapped_data))

#%%
#Creating new column using vectorized operation
df['new'] = df.apply(lambda x: x['col_a'] * x['col_b'], axis=1)

#%%
#remove some outliers. In the salary column, I want to keep 
#the values between the 5th and 95th quantiles.

low = np.quantile(marketing.Salary, 0.05)
high = np.quantile(marketing.Salary, 0.95)

df = df[df.Salary.between(low, high)]

#%%
#The dataframe contains many categorical variables. If the number of 
#categories are few compared to the total number values, it is better 
#to use the category data type instead of object. It saves a great 
#amount of memory depending on the data size.

#If the number of categories are less than 5 percent of the total
# number of values, the data type of the column will be changed
# to category.
     
cols = df.select_dtypes(include='object').columns

for col in cols:
    ratio = len(df[col].value_counts()) / len(df)
    if ratio < 0.05:
    df[col] = df[col].astype('category')

#%%
df.groupby('marital')['age'].describe()

#%%
df.groupby(['marital','response'])['age'].describe()
#%%
df.groupby(['marital','response'])['age'].mean()
#%%
df.groupby(['marital','response'])['age'].aggregate(np.sum)
#%%
df.groupby(['marital','response'])['age'].agg([np.sum, np.mean, np.std, len])

#%%
#data subsetting
df3.groupby(["marital"]).get_group("maried")
#%%


#%%
df.columns
#%%
df.age.aggregate({sum,min,max,len,np.mean})

#%%
df.aggregate({'Apps':['sum','min'],'Accept':'min'})
#%%
df[['Apps','Accept']].aggregate(['sum','min'])

#%%

#%%
#Lets first fix the university column name which is "Unnamed:0"
df.rename(columns={'Unnamed: 0':'univ_name'},inplace=True)

#%%
#%%
#%%
#%%
df.columns.tolist() #List of column names from df

#%%
#numeric columns names of data frame
numerics = ['int16', 'int32', 'int64', 'float64']
df.select_dtypes(include=numerics).columns

#%%
#numeric column names 
df.select_dtypes(include = [np.int64,np.float64]).columns.values.tolist()

#%%
#categorical column names
df.select_dtypes(include = [np.object]).columns.values.tolist()

#%%
#Subsetting Numerical data
numerical_data = df.select_dtypes(include = [np.int64,np.float64])

#%%
#Subsetting categorical data
numerical_data = df.select_dtypes(include = [np.object])





#%%
#%%
#%%

#%%
numeric_data = df.select_dtypes(include=np.number) # subsetting numeric data columns


#%%


#%%
#%%
#%%
# Checking the missing values counts in each column
df.isnull().sum()

#%%
#Checking the percentages of missing value in each column
(df.isnull().sum() / df.shape[0]) * 100

#%%
# Dropping the records with age missing in data dataframe.(droping all the age missing rows)
df2 = df[~df.month.isnull()].copy()
df2.isnull().sum()

#%%
#drop the records with response missing in data.
df3 = df2[~df2.response.isnull()].copy()
df3.isnull().sum()

#%%
# Let's see the null values in the month column.
df.month.isnull().sum()

#mode imputation for missing values in month column
month_mode = df.month.mode()[0]
month_mode
# Fill the missing values with mode value of month in data.
df.month.fillna(month_mode, inplace = True)

df.isnull().sum()

#%%
#%%
#%%
#subsetting all numeric columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df.select_dtypes(include=numerics)

#or    

neww_df = df.select_dtypes(include=np.number)
#%%
#If you want the names of numeric columns:
df.select_dtypes(include=np.number).columns.tolist()

#%%
# If your columns have numeric data but also have None, the dtype could be 
#'object'. This will coerce the columns to numeric: 
df.fillna(value=0, inplace=True) 


#%%
df.info()

#%%
#%%
#%%
#Subsetting data of one player
#df[df.Player == "Ersan Ilyasova"]

#%%
#%%
#%%
# Drop the customer id as it is of no use.
df.drop('customerid', axis = 1, inplace = True)

#%%
#%%
#%%
#create 2 columns from address clumn
#df[['city', 'country']] = df['address'].str.split(',', expand=True)

df.jobedu.value_counts()

df.jobedu.value_counts(normalize=True) 

#Extract job  & Education in newly from "jobedu" column.
df['job']= df["jobedu"].apply(lambda x: x.split(",")[0])

df['education']= df["jobedu"].apply(lambda x: x.split(",")[1])

#%%
df.info()

#%%
df.age.describe()

#%%
df.age.head(30)

#%%
# Change the data type

df["age"] = df['Customer Number'].astype('int')

df["Customer Number"] = df['Customer Number'].astype('str')

df['Is_Male'] = df.Is_Male.astype('category')

df["IsPurchased"] = df['IsPurchased'].astype('bool')

df["Total Spend"] = df['Total Spend'].astype('float')

df['Dates'] = pd.to_datetime(df['Dates'], format='%Y%m%d')


#%%

#9. See the data distribution and data anomaly
#From the summary statistics before, we might already know which 
#columns that potentially having data anomalies.

sns.displot(df, x="age")

#%%
#measure its skewness and kurtosis
df['age'].agg(['skew', 'kurtosis']).transpose()


#%%
#Let’s check the outlier for the total_bill column with Boxplot.
ax = sns.boxplot(x=df["age"])


#%%
#10. Check the correlation between variables in the data
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot = True, cmap= 'coolwarm')

#%%
#%%
#%%
#Handling Outliers
#-----------------

#Univariate outliers: Univariate outliers are the data points whose 
#values lie beyond the range of expected values based on one variable.

#Multivariate outliers: While plotting data, some values of one variable 
#may not lie beyond the expected range, but when you plot the data with 
#some other variable, these values may lie far from the expected value.

#So, after understanding the causes of these outliers, we can handle them by 
#dropping those records or imputing with the values or leaving them as is, if
# it makes more sense.


#Standardizing Values
#----------------------
#To perform data analysis on a set of values, we have to make sure the values in 
#the same column should be on the same scale. For example, if the data contains 
#the values of the top speed of different companies’ cars, then the whole column 
#should be either in meters/sec scale or miles/sec scale.

#%%
#%%
#%%
#3. Univariate Analysis
#----------------------
#Categorical Unordered Univariate Analysis:
#-------------------------------------------
#An unordered variable is a categorical variable that has no defined order. 
#If we take our data as an example, the job column in the dataset is divided 
#into many sub-categories like technician, blue-collar, services, management, etc. 
#There is no weight or measure given to any value in the ‘job’ column.

#Now, let’s analyze the job category by using plots. Since Job is a category, 
#we will plot the bar plot.

# Let's calculate the percentage of each job status category.
df.job.value_counts(normalize=True)

#plot the bar graph of percentage job categories
df.job.value_counts(normalize=True).plot.barh()
plt.show()

#%%
#Categorical Ordered Univariate Analysis:
#--------------------------------------------
#Ordered variables are those variables that have a natural rank of order. 
#Some examples of categorical ordered variables from our dataset are:

#Month: Jan, Feb, March……
#Education: Primary, Secondary,……

#Now, let’s analyze the Education Variable from the dataset. Since we’ve already 
#seen a bar plot, let’s see how a Pie Chart looks like.

#calculate the percentage of each education category.
df.education.value_counts(normalize=True)

#plot the pie chart of education categories
df.education.value_counts(normalize=True).plot.pie()
plt.show()

#%%
#Numeric variable
#--------------------
# If the column or variable is of numerical then we’ll analyze by calculating 
#its mean, median, std, etc. We can get those values by using the describe function.
df.salary.describe()

#%%
#%%
#%%
#4. Bivariate Analysis
#-----------------------
#If we analyze data by taking two variables/columns into consideration from a 
#dataset, it is known as Bivariate Analysis.

#a) Numeric-Numeric Analysis:
#----------------------------
# Scatter Plot, Pair Plot, Correlation Matrix

#1.Scatter Plot :Let’s take three columns ‘Balance’, ‘Age’ and ‘Salary’ from our dataset and 
#see what we can infer by plotting to scatter plot between salary balance and age balance
 
#plot the scatter plot of balance and salary variable in data
plt.scatter(df.salary,df.balance)
plt.show()


#plot the scatter plot of balance and age variable in data
df.plot.scatter(x="age",y="balance")
plt.show()   

#%%
#2. Pair Plot :   #plot the pair plot of salary, balance and age in data dataframe.
# We’ll use the seaborn library for plotting Pair Plots.

sns.pairplot(data = df, vars=['salary','balance','age'])
plt.show()

#%%
#3. Correlation Matrix : Since we cannot use more than two variables as x-axis 
#and y-axis in Scatter and Pair Plots, it is difficult to see the relation 
#between three numerical variables in a single graph. In those cases, we’ll use 
#the correlation matrix.

# Creating a matrix using age, salry, balance as rows and columns
df[['age','salary','balance']].corr()

#plot the correlation matrix of salary, balance and age in data dataframe.
sns.heatmap(df[['age','salary','balance']].corr(), annot=True, cmap = 'Reds')
plt.show()

#%%
#b) Numeric - Categorical Analysis
#--------------------------------------
#Analyzing the one numeric variable and one categorical variable from a dataset 
#is known as numeric-categorical analysis. We analyze them mainly using 
#mean, median, and box plots.

#Let’s take salary and response columns from our dataset.
#First check for mean value using groupby

##groupby the response to find the mean of the salary with response no & yes separately.
df.groupby('response')['salary'].mean()

#%%
#There is not much of a difference between the yes and no response based on the salary.
#Let’s calculate the median,
df.groupby('response')['salary'].median()


#%%
#By both mean and median we can say that the response of yes and no remains the
# same irrespective of the person’s salary. But, is it truly behaving like that,
# let’s plot the box plot for them and check the behavior.

#plot the box plot of salary for yes & no responses.
sns.boxplot(df.response, df.salary)
plt.show()

#As we can see, when we plot the Box Plot, it paints a very different picture 
#compared to mean and median. The IQR for customers who gave a positive response 
#is on the higher salary side.

#This is how we analyze Numeric-Categorical variables, we use mean, median, and 
#Box Plots to draw some sort of conclusions.

#%%
#c) Categorical — Categorical Analysis
#----------------------------------------
#Since our target variable/column is the Response rate, we’ll see how the 
#different categories like Education, Marital Status, etc., are associated with
# the Response column. So instead of ‘Yes’ and ‘No’ we will convert them into 
#‘1’ and ‘0’, by doing that we’ll get the “Response Rate”.

#create response_rate of numerical data type where response "yes"= 1, "no"= 0
df['response_rate'] = np.where(df.response=='yes',1,0)
df.response_rate.value_counts()

#%%
#Let’s see how the response rate varies for different categories in marital status.
#plot the bar graph of marital status with average value of response_rate
df.groupby('marital')['response_rate'].mean().plot.bar()
plt.show()

#Similarly, we can plot the graphs for Loan vs Response rate,
# Housing Loans vs Response rate, etc.

#%%
#%%
#%%
#5. Multivariate Analysis
#-----------------------------
#Let’s see how ‘Education’, ‘Marital’, and ‘Response_rate’ vary with each other.

#First, we’ll create a pivot table with the three columns and after that, 
#we’ll create a heatmap.

result = pd.pivot_table(data=df, index='education', columns='marital',values='response_rate')
print(result)

#create heat map of education vs marital vs response_rate
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

#Based on the Heatmap we can infer that the married people with primary education 
#are less likely to respond positively for the survey and single people with 
#tertiary education are most likely to respond positively to the survey.

#Similarly, we can plot the graphs for Job vs marital vs response, Education 
#vs poutcome vs response, etc.



#%%





































#%%
#%%
#%%

#------------------------------------------------------------------

#Map function that works as an iterator to return a result after applying a function to every item of an iterable object (tuple, lists, etc.).
#It is used when you want to apply a single transformation function to all the iterable elements. The iterable and function are passed as arguments to the map in Python


numbers = (1, 2, 3, 4)
result = map(lambda x: x + x, numbers)
print(list(result))




numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
result = map(lambda x, y: x + y, numbers1, numbers2)
print(list(result))




# List of strings
l = ['sat', 'bat', 'cat', 'mat']
# map() can listify the list of strings individually
test = list(map(list, l))
print(test)





# Return double of n
def addition(n):
    return n + n
# We double all numbers using map()
numbers = (1, 2, 3, 4)
result = map(addition, numbers)
print(list(result))






# returns square of a number
def square(number):
  return number * number

# apply square() function to each item of the numbers list
squared_numbers_iterator = list(map(square, numbers))






num1 = [4, 5, 6]
num2 = [5, 6, 7]

result = map(lambda n1, n2: n1+n2, num1, num2)

print(list(result))



#---------------------------------------------------------------------
#prime number is a  number that is divisible only by 1 and itself 
for n in range(2,100):
    for i in range(2,n):
        if(n%i==0):
            break
    else:
        print(n,end=' ') 


















